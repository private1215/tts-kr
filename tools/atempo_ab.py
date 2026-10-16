#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L1 快筛：现役「两次 atempo」是否制造「卡字/吃字」（2026-10-15 第30轮）

## 假设
fix_chain 现在对同一段音频做了**两次 WSOLA 时域拉伸**：
  ① 变速：每窗 `atempo=1/fs`（fs≈1.10，只放慢）
  ② 变调：`asetrate=sr*R, aresample=sr, atempo=1/R`（R≈0.92，降调）
两次叠加 → 语音被"揉"两遍。WSOLA 的典型听感就是**起音被抹、辅音糊掉、
过渡发硬**，和员工说的「吃字 / 卡字」高度吻合。

而这两步在数学上可以**合并成一次**：
  asetrate=sr*R, aresample=sr, atempo=1/(R*fs)
  → 时长 orig/R × (R·fs) = orig·fs ✓   音高 ×R ✓   **只走一遍 WSOLA**

## 实验
拿 EL 真人标杆（真实语音、本来没 artifacts）当前后一致的素材，
用两种写法处理到**完全相同的时长和音高**，比谁把素材伤得更重：
  A 现役（两次 atempo）  B 融合（一次 atempo）  P 只变调（不变速）
指标：平坦度 / 4k~8k 占比 / 起音数 / 起音强度 / 谱通量突变数

## 用法
    ./venv/bin/python tools/atempo_ab.py [--sec 120] [--fs 1.10] [--r 0.92]
"""
import argparse
import json
import os
import subprocess
import tempfile

import numpy as np

SR = 16000
DIAG = "/mnt/tts/diag/muffle"
OUT = os.path.join(DIAG, "atempo_ab.json")
EL = os.path.join(DIAG, "el24.wav")

try:
    import librosa
except Exception:
    librosa = None


def load16(p):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-ac", "1",
                          "-ar", str(SR), "-f", "f32le", "-"],
                         stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def ff(src, dst, af, sr_out=None):
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", src, "-af", af]
    if sr_out:
        cmd += ["-ar", str(sr_out)]
    cmd += ["-f", "wav", dst]
    subprocess.run(cmd, check=True)


def flat_pg(y16):
    if librosa is None:
        return None
    return float(librosa.feature.spectral_flatness(
        y=y16[:60 * SR], hop_length=256)[0].mean())


def hf_ratio(y16):
    n, hop = 2048, 512
    nfr = (len(y16) - n) // hop
    idx = np.arange(nfr)[:, None] * hop + np.arange(n)[None, :]
    fr = y16[idx] * np.hanning(n)[None, :]
    sp = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1.0 / SR)
    tot = sp.sum()
    return float(sp[:, (f >= 4000) & (f < 8000)].sum() / tot * 100)


def onsets(y16):
    """起音数 / 平均起音强度：WSOLA 会抹平起音 → 数变少、强度变弱。"""
    if librosa is None:
        return None, None
    env = librosa.onset.onset_strength(y=y16, sr=SR, hop_length=256)
    pk = librosa.util.peak_pick(env, pre_max=3, post_max=3,
                                pre_avg=3, post_avg=5, delta=0.10, wait=5)
    return float(len(pk)), float(env.mean())


def flux_jumps(y16):
    """谱通量突变数：帧间谱变化超过 8×MAD 的次数（突变 = 听感硬/卡）。"""
    n, hop = 1024, 256
    nfr = (len(y16) - n) // hop
    idx = np.arange(nfr)[:, None] * hop + np.arange(n)[None, :]
    fr = y16[idx] * np.hanning(n)[None, :]
    sp = np.abs(np.fft.rfft(fr, axis=1))
    fl = np.sqrt(((np.diff(sp, axis=0)) ** 2).sum(axis=1))
    med = np.median(fl)
    mad = np.median(np.abs(fl - med)) * 1.4826 + 1e-12
    return int(((fl - med) > 8 * mad).sum())


def measure(tag, y, dur):
    r = {"tag": tag, "dur": dur,
         "flat": flat_pg(y), "hf4_8": hf_ratio(y), "flux": flux_jumps(y)}
    n, s = onsets(y)
    r["onsets"], r["onset_str"] = n, s
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sec", type=float, default=120.0)
    ap.add_argument("--fs", type=float, default=1.10)   # 需要放慢的倍数
    ap.add_argument("--r", type=float, default=0.92)    # 变调比例
    a = ap.parse_args()

    d = tempfile.mkdtemp(dir="/mnt/tts/diag")
    head = os.path.join(d, "head.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", EL,
                    "-t", str(a.sec), "-ar", str(SR), "-ac", "1",
                    "-f", "wav", head], check=True)
    base = load16(head)
    rows = [measure("EL 原样", base, len(base) / SR)]

    sr = SR
    # A 现役：先 atempo=1/fs，再 asetrate+aresample+atempo=1/R
    p1 = os.path.join(d, "a1.wav")
    p2 = os.path.join(d, "a2.wav")
    ff(head, p1, "atempo=%.6f" % (1.0 / a.fs))
    ff(p1, p2, "asetrate=%d,aresample=%d,atempo=%.6f"
       % (int(round(sr * a.r)), sr, 1.0 / a.r))
    rows.append(measure("A 现役(2次atempo)", load16(p2),
                        len(load16(p2)) / SR))

    # B 融合：一次 asetrate+aresample + 一次 atempo=1/(R*fs)
    p3 = os.path.join(d, "b1.wav")
    ff(head, p3, "asetrate=%d,aresample=%d,atempo=%.6f"
       % (int(round(sr * a.r)), sr, 1.0 / (a.r * a.fs)))
    rows.append(measure("B 融合(1次atempo)", load16(p3),
                        len(load16(p3)) / SR))

    # P 只变调：asetrate+aresample+atempo=1/R（不变速）
    p4 = os.path.join(d, "p1.wav")
    ff(head, p4, "asetrate=%d,aresample=%d,atempo=%.6f"
       % (int(round(sr * a.r)), sr, 1.0 / a.r))
    rows.append(measure("P 只变调", load16(p4), len(load16(p4)) / SR))

    print("\n%-20s %8s %9s %8s %8s %8s %9s"
          % ("版本", "时长s", "平坦度", "4k~8k%", "起音数", "起音强", "谱突变"))
    for r in rows:
        print("%-20s %8.1f %9.5f %8.4f %8.0f %8.3f %9d"
              % (r["tag"], r["dur"], r["flat"], r["hf4_8"],
                 r["onsets"], r["onset_str"], r["flux"]))

    print("\n时长应 = %.1fs（= %.1f × %.2f）" %
          (a.sec * a.fs, a.sec, a.fs))
    print("A 时长 %.2f  B 时长 %.2f  → 一致性 %s"
          % (rows[1]["dur"], rows[2]["dur"],
             "OK" if abs(rows[1]["dur"] - rows[2]["dur"]) < 0.3 else "❌不一致"))
    print("平坦度：EL %.5f → A %.5f / B %.5f   （B 越接近 EL 越好）"
          % (rows[0]["flat"], rows[1]["flat"], rows[2]["flat"]))
    print("4k~8k ：EL %.4f → A %.4f / B %.4f"
          % (rows[0]["hf4_8"], rows[1]["hf4_8"], rows[2]["hf4_8"]))

    json.dump(rows, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("\n→ " + OUT)


if __name__ == "__main__":
    main()
