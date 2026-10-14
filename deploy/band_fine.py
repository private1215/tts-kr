#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""细分频段诊断：员工说「声音像感冒后发出的声音」。

感冒音 = 鼻音重 / 低频混浊 / 不通透。粗粒度的 100Hz-1k 已经对齐了 EL，
所以要看**细分**：是不是 100~400Hz 这一坨（鼻音共振区）明显偏多。
"""
import subprocess
import tempfile
import os
import numpy as np
import soundfile as sf
import librosa

SR = 24000
DUR = 60.0
BANDS = [(60, 150), (150, 300), (300, 500), (500, 1000),
         (1000, 2000), (2000, 4000), (4000, 8000)]

FILES = [
    ("EL案例8197", "/mnt/tts/data/cases/8197d9ac8a35/merged.wav"),
    ("EL案例1a6c", "/mnt/tts/data/cases/1a6c20a60481/merged.wav"),
    ("Vox raw", "/mnt/tts/diag/x8197/vox_male_raw.wav"),
    ("nof0(清洗+EQ未变调)", "/mnt/tts/diag/x8197/m_nof0.wav"),
    ("base_p 全局变调裸片", "/mnt/tts/diag/x8197/base_p.wav"),
    ("base_pw 分窗变调裸片", "/mnt/tts/diag/x8197/base_pw.wav"),
    ("v4(旧EQ)", "/mnt/tts/diag/x8197/m_v4.wav"),
    ("v5(新EQ)", "/mnt/tts/diag/x8197/m_v5.wav"),
    ("v5-cf9", "/mnt/tts/diag/cf9/m_v5.wav"),
]


def bands(path):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", str(SR), "-ac", "1", t], check=True)
    y, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    y = y[:int(DUR * SR)]
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    f = librosa.fft_frequencies(sr=SR, n_fft=2048)
    tot = S.sum() + 1e-12
    return [float(S[(f >= lo) & (f < hi)].sum() / tot) for lo, hi in BANDS]


if __name__ == "__main__":
    hdr = "文件\t" + "\t".join("%d-%d" % b for b in BANDS)
    print(hdr)
    for n, p in FILES:
        try:
            v = bands(p)
            print(n + "\t" + "\t".join("%.4f" % x for x in v))
        except Exception as e:
            print("FAIL", n, repr(e)[:100])
