# -*- coding: utf-8 -*-
"""LTAS 差异曲线定位：看到底是哪个频段差、差多少。"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.fidelity import load  # noqa: E402
import librosa  # noqa: E402

CASE = "/mnt/tts/data/cases/1a6c20a60481/merged.wav"
PROD = "/tmp/eval5work/1a6c20.wav"


def ltas(x, sr, n_fft=2048, hop=512, n_mels=128, top=50):
    S = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop)) ** 2
    fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=50, fmax=sr / 2)
    mf = librosa.mel_frequencies(n_mels=n_mels, fmin=50, fmax=sr / 2)
    M = fb @ S
    e = M.mean(axis=0)
    sel = e > np.percentile(e, top)
    p = M[:, sel].mean(axis=1)
    return 10 * np.log10(p / p.sum() + 1e-12), mf


def main():
    for sr0 in (24000, 44100):
        xc, _ = load(CASE, sr0=sr0)
        xp, _ = load(PROD, sr0=sr0)
        a, fa = ltas(xc, sr0)
        b, fb = ltas(xp, sr0)
        a = a - a.mean()
        b = b - b.mean()
        print(f"\n===== sr={sr0} =====")
        print("mel 中心频率(抽样) / 案例 / 成品 / 差")
        for i in range(0, len(fa), 8):
            print(f"  {fa[i]:8.0f}  {a[i]:7.2f} {b[i]:7.2f}  {b[i]-a[i]:+7.2f}")
        m = fa <= 11000
        print(f"  <11k 平均|差| = {np.mean(np.abs((b-a)[m])):.3f} dB  "
              f"带符号均值 = {np.mean((b-a)[m]):+.3f} dB")
        # 1/3 倍频程聚合
        import math
        bands = [65, 92, 131, 185, 262, 370, 523, 740, 1047, 1480, 2093,
                 2960, 4186, 5920, 8372]
        devs = []
        for lo_, hi_ in zip(bands, bands[1:] + [11000]):
            sel = (fa >= lo_) & (fa < hi_)
            if sel.sum() == 0:
                continue
            devs.append((lo_, float(np.mean(b[sel]) - np.mean(a[sel]))))
        print("  1/3oct 带符号差：")
        print("   " + "  ".join(f"{int(f)}:{d:+.1f}" for f, d in devs))
        print(f"  1/3oct 平均|差| = {np.mean([abs(d) for _, d in devs]):.3f} dB")


if __name__ == "__main__":
    main()
