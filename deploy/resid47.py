#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""resid47.py —— step1(fix_chain) vs r_vol(纯重采样+同增益) 的波形残差。

两者理论上应该**逐样本相同**（都 = 裸输出 -> 16k -> +1.4dB，无 EQ/clean/pitch）。
若残差是宽带噪声 → fix_chain 有实质损伤；若残差≈0 → 只是增益差。
"""
import subprocess

import numpy as np

SR = 16000


def load(p):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-ar", str(SR),
                        "-ac", "1", "-f", "f32le", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.float32)


def main():
    a = load("/mnt/tts/diag/raw47/r_vol.wav")
    b = load("/mnt/tts/diag/raw47/step1.wav")
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    print(f"长度 {len(a)/SR:.2f}s / {len(b)/SR:.2f}s")

    # 最佳增益对齐
    g = float((a @ b) / (a @ a))
    print(f"最佳增益 b/a = {g:.6f}  ({20*np.log10(g):+.3f} dB)")

    res = b - g * a
    snr = 10 * np.log10((b @ b) / max(res @ res, 1e-30))
    print(f"残差 SNR = {snr:.2f} dB   "
          f"(残差 RMS {np.sqrt((res**2).mean()):.6f}, "
          f"信号 RMS {np.sqrt((b**2).mean()):.6f})")

    # 残差集中在哪：按 1s 窗看残差能量 top10
    w = SR
    m = (len(res) // w) * w
    e = np.sqrt((res[:m].reshape(-1, w) ** 2).mean(axis=1))
    top = np.argsort(e)[::-1][:10]
    print("\n残差能量 top10（这些秒 fix_chain 动得最多）:")
    for i in sorted(top):
        print(f"  {i:6d}s  残差RMS={e[i]:.6f}")

    # 残差的频谱质心：宽带=损伤，低频=增益/滤波差异
    X = np.abs(np.fft.rfft(res * np.hanning(len(res)))) ** 2
    f = np.fft.rfftfreq(len(res), 1 / SR)
    cen = float((X * f).sum() / X.sum())
    print(f"\n残差频谱质心 = {cen:.0f} Hz")
    tot = X.sum()
    for lo, hi in [(0, 300), (300, 1000), (1000, 3000), (3000, 8000)]:
        msk = (f >= lo) & (f < hi)
        print(f"  {lo:5d}-{hi:5d}Hz  {X[msk].sum()/tot*100:6.1f}%")


if __name__ == "__main__":
    main()
