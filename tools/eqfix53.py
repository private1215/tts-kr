# -*- coding: utf-8 -*-
"""eqfix53.py —— 用 firequalizer 扫描中频增益，对齐 EL 音色。

## 为什么换 firequalizer

本机 ffmpeg 的 `equalizer` 滤镜实测**失效**（+8dB 后 1k-3k 占比 17.36→17.40，不动）。
`firequalizer`（FIR）实测有效（+5dB → 17.36→28.11）。

## 目标（measure53.py 同文本 EL 对照）

    发声帧中频1k-3k占比：我们 5.76% vs EL 6.86%
    click%            ：我们 0.0172 vs EL 0.0124
    静音杂音          ：我们 4.2%  vs EL 5.0%

## 做法

gain_entry 曲线：1k 处 0dB、1.8k 处 +G dB、3k 处 0dB（3k 以上保持 0，防嘶声）。
扫描 G，选「中频最接近 EL 且 click/杂音不恶化」的点，输出成品。

用法：python3 eqfix53.py <ours.wav> <el.wav> <out.wav>
"""
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 0.005
VTH = -40.0


def load16(p):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(SR), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    return x.astype(np.float64)


def eq(p, g):
    if g == 0:
        return p
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p, "-af",
                    "firequalizer=gain_entry='entry(1000,0);"
                    "entry(1800,%g);entry(3000,0)'" % g,
                    "-ar", str(SR), "-ac", "1", t], check=True)
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
    if len(sys.argv) < 4:
        print("用法：eqfix53.py <ours.wav> <el.wav> <out.wav>")
        return 1
    ours, el, out = sys.argv[1:4]
    e = metrics(el)
    print("EL 目标：中频 %.2f%%  质心 %.0f  click %.4f%%  杂音 %.2f%%"
          % (e["mid"], e["centroid"], e["click"], e["noise"]))
    print()
    print("%-8s %8s %8s %10s %8s" % ("增益dB", "中频%", "质心Hz",
                                     "click%", "杂音%"))
    best = None
    for g in (0, 1, 2, 3, 4, 5):
        m = metrics(eq(ours, g))
        print("%-8.1f %8.2f %8.0f %10.4f %8.2f"
              % (g, m["mid"], m["centroid"], m["click"], m["noise"]))
        err = abs(m["mid"] - e["mid"]) / e["mid"]
        pen = 0.0
        if m["click"] > e["click"] * 1.5:
            pen += 1.0
        if m["noise"] > e["noise"] * 1.5:
            pen += 1.0
        sc = err + pen
        if best is None or sc < best[0]:
            best = (sc, g, m)
    g = best[1]
    print("\n最佳增益 = %g dB → 中频 %.2f%%（EL %.2f%%）click %.4f%% 杂音 %.2f%%"
          % (g, best[2]["mid"], e["mid"], best[2]["click"], best[2]["noise"]))
    tmp = eq(ours, g)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp,
                    "-ar", str(SR), "-ac", "1", out], check=True)
    print("-> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
