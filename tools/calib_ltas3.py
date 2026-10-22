# -*- coding: utf-8 -*-
"""LTAS 口径校准 3：按 docs/r13 §7 的描述逐条还原。

    两边重采样到 24 kHz、只比 60Hz~11kHz、只取有声帧（帧 RMS 高于该段 p35）、
    **每段先按总能量归一化**，然后逐段平均。

目标：标杆男 r20 成品 vs 案例 ≈ 0.36 dB。
"""
import os
import sys
import numpy as np
import librosa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.fidelity import load, sent_windows  # noqa: E402

CASE = "/mnt/tts/data/cases/1a6c20a60481/merged.wav"
PROD = "/tmp/eval5work/1a6c20.wav"
N_FFT, HOP, N_MELS = 2048, 512, 128
BANDS = [65, 92, 131, 185, 262, 370, 523, 740, 1047, 1480, 2093, 2960,
         4186, 5920, 8372]


def _mel(sr):
    return (librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=N_MELS,
                                fmin=50, fmax=sr / 2),
            librosa.mel_frequencies(n_mels=N_MELS, fmin=50, fmax=sr / 2))


def seg_ltas(x, sr, segs, pct=35, limit=80):
    fb, mf = _mel(sr)
    out = []
    for s, e in segs[:limit]:
        y = x[s:e]
        if len(y) < N_FFT * 2:
            continue
        S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2
        M = fb @ S
        fe = M.mean(axis=0)
        sel = fe > np.percentile(fe, pct)
        if sel.sum() < 3:
            continue
        p = M[:, sel].mean(axis=1)
        out.append(10 * np.log10(p / (p.sum() + 1e-20) + 1e-12))
    if not out:
        return None, mf
    return np.mean(np.array(out), axis=0), mf


def global_ltas(x, sr, pct=35):
    fb, mf = _mel(sr)
    S = np.abs(librosa.stft(x, n_fft=N_FFT, hop_length=HOP)) ** 2
    M = fb @ S
    fe = M.mean(axis=0)
    sel = fe > np.percentile(fe, pct)
    p = M[:, sel].mean(axis=1)
    return 10 * np.log10(p / (p.sum() + 1e-20) + 1e-12), mf


def band_dev(a, b, fa, fb_):
    """1/3 倍频程（= EQ 的 15 段）上的平均绝对偏差 dB。"""
    devs = []
    for lo, hi in zip(BANDS, BANDS[1:] + [11000]):
        sa = (fa >= lo) & (fa < hi)
        sb = (fb_ >= lo) & (fb_ < hi)
        if sa.sum() == 0 or sb.sum() == 0:
            continue
        devs.append(float(np.mean(a[sa]) - np.mean(b[sb])))
    return devs


def mel_dev(a, b, m):
    return float(np.mean(np.abs((a - b)[m])))


def main():
    xc, sr = load(CASE, sr0=24000)
    xp, _ = load(PROD, sr0=24000)
    segs_c, _, _ = sent_windows(xc, sr)
    segs_p, _, _ = sent_windows(xp, sr)
    m = (librosa.mel_frequencies(n_mels=N_MELS, fmin=50, fmax=12000) <= 11000)

    for tag, fn in (("全局 p35", "g35"), ("全局 p50", "g50"),
                    ("逐段 p35", "s35"), ("逐段 p50", "s50")):
        if fn.startswith("g"):
            pct = 35 if fn == "g35" else 50
            a, fa = global_ltas(xc, sr, pct)
            b, fb_ = global_ltas(xp, sr, pct)
        else:
            pct = 35 if fn == "s35" else 50
            a, fa = seg_ltas(xc, sr, segs_c, pct)
            b, fb_ = seg_ltas(xp, sr, segs_p, pct)
        devs = band_dev(a, b, fa, fb_)
        print(f"{tag}: mel平均|差| {mel_dev(a, b, m):.3f} dB | "
              f"1/3oct平均|差| {np.mean(np.abs(devs)):.3f} dB")
        print("   1/3oct 逐段: " + " ".join(f"{d:+.1f}" for d in devs))


if __name__ == "__main__":
    main()
