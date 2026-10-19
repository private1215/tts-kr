# -*- coding: utf-8 -*-
"""eqfix53b.py —— eqfix53 的**保采样率**版（只动 1k-3k 窄带，不做全频段 LTAS）。

## 为什么要有 53b（2026-10-19 实测）
pipe_1019e 跑完整链（**全频段 LTAS** → 53b2 → 53c2 → clickfix53b）结果：

    中频 0.84 → 1.05 ✅   （目标达成）
    click 0.05× → **1.72×** ❌  （反而比 EL 差 72%）

LTAS 要对全频段做多段 FIR，振铃/时域涂抹量远大于单条窄带 EQ → 引入 click。
而现役 v3.1 的 click 是 **0.05×EL**（比 EL 好 20 倍），是**已经比 EL 好的指标**，
按决策铁律「已比 EL 好的不动」——**不应该为了中频把它毁掉**。

→ 改用 eqfix53 的窄带方案（1k 处 0dB / 1.8k 处 +G / 3k 处 0dB），FIR 阶数效应小得多。
记忆里这条已验证有效：「男声中频 EQ +2dB：5.76% → 6.92%（EL 6.86%）」。

## 与 53 版的差别
- `eq()` 施加 firequalizer **不再带 `-ar 16000`**
- 最终写出用**原生采样率**（不是 16 kHz）
- 测量仍在 16k 做（那是口径，不能改）
- 新增 `--gain` 强制指定增益（不给就自动扫描 0~5dB）
- 新增 `--mp3` 顺便压一份 mp3，用来按铁律测「压完 mp3 后的 click」

🔴 本机 ffmpeg `equalizer` 失效，必须用 firequalizer（FIR）。

用法: python3 eqfix53b.py <ours.wav> <el.wav> <out.wav> [--gain 2] [--mp3]
"""
import argparse
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000          # 仅用于**测量**
FRAME = 0.005
VTH = -40.0


def native_sr(p):
    import json
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                        "-show_entries", "stream=sample_rate",
                        "-of", "json", p], capture_output=True, text=True)
    return int(json.loads(o.stdout)["streams"][0]["sample_rate"])


def load16(p):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(SR), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    return x.astype(np.float64)


def eq(p, g):
    """🔴 保采样率：firequalizer 后不带 -ar 16000"""
    if g == 0:
        return p
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p, "-af",
                    "firequalizer=gain_entry='entry(1000,0);"
                    "entry(1800,%g);entry(3000,0)'" % g,
                    "-ac", "1", t], check=True)
    return t


def metrics(p):
    x = load16(p)
    pk = np.max(np.abs(x))
    xn = x * (0.70 / pk) if pk > 0 else x
    click = float((np.abs(np.diff(xn)) >= 0.30).mean())
    n = int(FRAME * SR)
    nfr = len(x) // n
    fr = x[:nfr * n].reshape(nfr, n)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-12)
    v = db > VTH
    sp = np.abs(np.fft.rfft(fr * np.hanning(n), axis=1)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    mid, cen, noi = [], [], []
    for i in range(len(v)):
        s = sp[i]
        tot = s.sum() + 1e-12
        if v[i]:
            mid.append(float(s[(freqs >= 1000) & (freqs <= 3000)].sum() / tot))
            cen.append(float((freqs * s).sum() / tot))
        else:
            noi.append(float(s[(freqs >= 4000) & (freqs <= 8000)].sum() / tot))
    return dict(click=click * 100,
                mid=float(np.median(mid)) * 100 if mid else 0,
                centroid=float(np.median(cen)) if cen else 0,
                noise=float(np.median(noi)) * 100 if noi else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ours")
    ap.add_argument("el")
    ap.add_argument("out")
    ap.add_argument("--gain", type=float, default=None,
                    help="强制增益 dB（不给则扫描 0~5）")
    ap.add_argument("--mp3", action="store_true", help="顺便压 mp3")
    a = ap.parse_args()

    nsr = native_sr(a.ours)
    print("原生采样率 %d Hz（成品将保持此采样率）" % nsr, flush=True)
    e = metrics(a.el)
    print("EL 目标：中频 %.2f%%  质心 %.0f  click %.4f%%  杂音 %.2f%%"
          % (e["mid"], e["centroid"], e["click"], e["noise"]))
    print()
    print("%-8s %8s %8s %10s %8s" % ("增益dB", "中频%", "质心Hz",
                                     "click%", "杂音%"))
    gains = (a.gain,) if a.gain is not None else (0, 1, 2, 3, 4, 5)
    best = None
    for g in gains:
        m = metrics(eq(a.ours, g))
        print("%-8.1f %8.2f %8.0f %10.4f %8.2f"
              % (g, m["mid"], m["centroid"], m["click"], m["noise"]))
        err = abs(m["mid"] - e["mid"]) / e["mid"]
        pen = 0.0
        # 🔴 比 eqfix53 更严：click/杂音**只要比 EL 差就罚**（现役本来就比 EL 好）
        if m["click"] > e["click"]:
            pen += 2.0 * (m["click"] / max(e["click"], 1e-9) - 1.0)
        if m["noise"] > e["noise"]:
            pen += 0.5 * (m["noise"] / max(e["noise"], 1e-9) - 1.0)
        sc = err + pen
        if best is None or sc < best[0]:
            best = (sc, g, m)
    g = best[1]
    print("\n最佳增益 = %g dB → 中频 %.2f%%（EL %.2f%%）click %.4f%% 杂音 %.2f%%"
          % (g, best[2]["mid"], e["mid"], best[2]["click"], best[2]["noise"]))

    tmp = eq(a.ours, g)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp,
                    "-ar", str(nsr), "-ac", "1", a.out], check=True)
    print("-> %s  @ %d Hz" % (a.out, nsr))
    if a.mp3:
        mp3 = a.out.rsplit(".", 1)[0] + ".mp3"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", a.out,
                        "-codec:a", "libmp3lame", "-b:a", "192k", mp3],
                       check=True)
        print("-> %s" % mp3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
