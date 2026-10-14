#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全篇频段对比。

🔴 band_fine.py 只看**前 60s**，而变调是**分窗**的：第一窗变调量最大
   （1.220），后面逐窗降到 1.045。拿前 60s 比，等于只看了最坏的那一段，
   分窗变调的收益完全被掩盖（base_p 与 base_pw 前 60s 几乎一样）。
   要看真实分布必须测全篇。
"""
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import librosa

SR = 24000
BANDS = [(60, 150), (150, 300), (300, 500), (500, 1000),
         (1000, 2000), (2000, 4000), (4000, 8000)]

FILES = [
    ("EL 8197", "/mnt/tts/data/cases/8197d9ac8a35/merged.wav"),
    ("EL 1a6c", "/mnt/tts/data/cases/1a6c20a60481/merged.wav"),
    ("Vox raw", "/mnt/tts/diag/x8197/vox_male_raw.wav"),
    ("base_p 全局变调", "/mnt/tts/diag/x8197/base_p.wav"),
    ("base_pw 分窗变调", "/mnt/tts/diag/x8197/base_pw.wav"),
    ("v5 旧(全局变调)", "/mnt/tts/diag/x8197/m_v5.wav"),
    ("v7 新(分窗变调)", "/mnt/tts/diag/x8197/m_v7.wav"),
]


def bands(path):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", str(SR), "-ac", "1", t], check=True)
    y, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    f = librosa.fft_frequencies(sr=SR, n_fft=2048)
    tot = S.sum() + 1e-12
    out = [float(S[(f >= lo) & (f < hi)].sum() / tot) for lo, hi in BANDS]
    os.environ.setdefault("X", "")
    return out, len(y) / SR


if __name__ == "__main__":
    print("文件\t时长\t" + "\t".join("%d-%d" % b for b in BANDS))
    for n, p in FILES:
        if not os.path.exists(p):
            print(n, "缺失")
            continue
        v, d = bands(p)
        print("%-16s %.0fs\t" % (n, d) + "\t".join("%.4f" % x for x in v))
