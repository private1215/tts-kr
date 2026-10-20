# -*- coding: utf-8 -*-
"""head_harm.py —— 判定文件头 0.07~0.28s 的突起是「浊音（有谐波）」还是「纯蜂音」，
并和模版 / EL 标杆的头部做同口径对照。"""
import os
import subprocess

import numpy as np
import soundfile as sf

WORK = "/mnt/tts/diag/v0902"
SR = 16000
FILES = [
    ("我们 202610190902", "/mnt/tts/output/202610190902/202610190902-1.mp3"),
    ("模版 EL_female", "/mnt/tts/data/voices/EL_female.wav"),
    ("EL 标杆 00b6a45", "/mnt/tts/data/cases/00b6a45ad6aa/merged.wav"),
]


def to16k(path, out):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "16000", "-ac", "1", out], check=True)
    return out


def head_env(x, sr, n=1.6):
    hop = int(0.01 * sr)
    vals = []
    for i in range(int(n / 0.01)):
        seg = x[i * hop:(i + 1) * hop]
        vals.append(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12)
                    if len(seg) else -120)
    return vals


def main():
    for name, p in FILES:
        wav = os.path.join(WORK, os.path.basename(p) + ".16k.wav")
        if not os.path.exists(wav):
            to16k(p, wav)
        x, sr = sf.read(wav, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        v = head_env(x, sr)
        print("\n=== %s ===" % name)
        print("  头部 10ms 包络 dBFS（0~1.6s）：")
        for k in range(0, len(v), 60):
            print("   %5.2fs " % (k * 0.01) +
                  " ".join("%5.0f" % q for q in v[k:k + 60]))
        # 前 0.4s 的谐波结构
        s = x[int(0.05 * sr):int(0.30 * sr)]
        if len(s) > 256:
            S = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
            fr = np.fft.rfftfreq(len(s), 1.0 / sr)
            idx = np.argsort(S)[::-1][:8]
            pk = sorted([(round(float(fr[i]), 1),
                          round(float(10 * np.log10(S[i] + 1e-12)), 1))
                         for i in idx])
            print("  0.05~0.30s 主要谱峰 (Hz, dB): %s" % pk)
            # 自相关估 f0
            s2 = s - s.mean()
            ac = np.correlate(s2, s2, "full")[len(s2) - 1:]
            ac /= (ac[0] + 1e-12)
            lo, hi = int(0.003 * sr), int(0.015 * sr)   # 66~333Hz
            k = lo + int(np.argmax(ac[lo:hi]))
            print("  自相关 f0 ≈ %.1f Hz（峰高 %.2f）" % (sr / k, ac[k]))


if __name__ == "__main__":
    main()
