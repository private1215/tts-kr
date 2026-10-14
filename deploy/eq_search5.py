#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分窗变调后的 EQ 网格搜索（第五轮，全篇口径）。

🔴 为什么不能再拿前 60s 搜：
   变调现在是**分窗**的，第一窗比例最大（1.220）、后面降到 1.045。
   前 60s 恰好是最坏那一窗，用它搜出来的 EQ 会**过度补偿**整篇
   （band_full.py 实测：前60s 500-1k 0.2038 / 全篇只有 0.1777）。
   所以本轮统一取 **[60s, 360s] 这 300 秒** 的中段来量，EL 也取同一段。

优化：源与 EL 都先裁好再进循环，每组只跑 1 次 ffmpeg + 1 次 STFT（≈3s）。
"""
import itertools
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import librosa

SR = 24000
START = float(os.environ.get("BAND_START", "60"))
DUR = float(os.environ.get("BAND_DUR", "300"))
BANDS = [(60, 150), (150, 300), (300, 500), (500, 1000),
         (1000, 2000), (2000, 4000), (4000, 8000)]

SRC = os.environ.get("SRC", "/mnt/tts/diag/x8197/base_pw.wav")
ELS = ["/mnt/tts/data/cases/8197d9ac8a35/merged.wav",
       "/mnt/tts/data/cases/1a6c20a60481/merged.wav"]


def trim24(path, out):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(START),
                    "-t", str(DUR), "-i", path, "-ar", str(SR), "-ac", "1",
                    out], check=True)
    y, _ = sf.read(out, dtype="float32")
    return y


def bands_arr(y):
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    f = librosa.fft_frequencies(sr=SR, n_fft=2048)
    tot = S.sum() + 1e-12
    return np.array([float(S[(f >= lo) & (f < hi)].sum() / tot)
                     for lo, hi in BANDS])


def flat_of(y):
    return float(librosa.feature.spectral_flatness(y=y, hop_length=256)[0].mean())


if __name__ == "__main__":
    d = tempfile.mkdtemp()
    src_t = os.path.join(d, "src.wav")
    y_src = trim24(SRC, src_t)
    tgt = np.mean([bands_arr(trim24(p, os.path.join(d, "el%d.wav" % i)))
                   for i, p in enumerate(ELS)], axis=0)
    tflat = np.mean([flat_of(trim24(p, os.path.join(d, "f%d.wav" % i)))
                     for i, p in enumerate(ELS)])
    b0 = bands_arr(y_src)
    print("口径: [%.0fs, %.0fs] @24k" % (START, START + DUR))
    print("EL目标\t" + "\t".join("%.4f" % x for x in tgt)
          + "\tflat=%.5f" % tflat)
    print("裸片\t" + "\t".join("%.4f" % x for x in b0)
          + "\tL1=%.4f\tflat=%.5f" % (np.abs(b0 - tgt).sum(), flat_of(y_src)))

    rows = []
    for A, B, C, D2, E in itertools.product([0, 3, 6],
                                            [1, 2, 3],
                                            [1.5, 2.5],
                                            [0, 1.5, 3.0],
                                            [6.5, 9.5]):
        parts = []
        if A:
            parts.append("equalizer=f=100:t=q:w=1.0:g=%d" % A)
        parts.append("equalizer=f=800:t=q:w=1.2:g=%d" % -B)
        parts.append("highshelf=f=1500:g=%g" % C)
        if D2:
            parts.append("highshelf=f=2500:g=%g" % -D2)
        parts.append("highshelf=f=4000:g=%g" % -E)
        parts.append("lowpass=f=14000")
        af = ",".join(parts)
        o = os.path.join(d, "o.wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src_t,
                        "-af", af, o], check=True)
        y, _ = sf.read(o, dtype="float32")
        b = bands_arr(y)
        rows.append((float(np.abs(b - tgt).sum()), A, B, C, D2, E,
                     flat_of(y), b, af))
    rows.sort(key=lambda r: r[0])
    print("\nL1\t低补\t鼻衰\t1.5k\t2.5k\t4k\tflat\t" + "\t".join(
        "%d-%d" % x for x in BANDS))
    for r in rows[:15]:
        print("%.4f\t+%d\t-%d\t+%.1f\t-%.1f\t-%.1f\t%.5f\t"
              % (r[0], r[1], r[2], r[3], r[4], r[5], r[6])
              + "\t".join("%.4f" % x for x in r[7]))
    print("\nBEST:", rows[0][8])
