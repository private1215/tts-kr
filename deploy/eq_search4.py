#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""窄带补低频专项（第四轮）。

前三轮的死结：60-150Hz（胸腔）补不上来。
  · lowshelf=f=150 过渡带太宽，连带抬 150~500（那两段我们并不缺）→ L1 反而变差
  · equalizer=f=100:w=1.0 也一样，网格里 +3 就把 L1 从 0.0934 推到 0.1202
本轮用**更窄**（w=0.5~0.7 倍频程）+ 更大增益，看能不能只动 60-150 这一格。
固定其余为第三轮最优：鼻音 -2 / 1.5k +1.5 / 4k -9.5 / lowpass 14000。
"""
import itertools
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import librosa

SR = 24000
DUR = 60.0
BANDS = [(60, 150), (150, 300), (300, 500), (500, 1000),
         (1000, 2000), (2000, 4000), (4000, 8000)]

SRC = os.environ.get("SRC", "/mnt/tts/diag/x8197/base_p.wav")
ELS = ["/mnt/tts/data/cases/8197d9ac8a35/merged.wav",
       "/mnt/tts/data/cases/1a6c20a60481/merged.wav"]
FIX = "equalizer=f=800:t=q:w=1.2:g=-2,highshelf=f=1500:g=1.5,highshelf=f=4000:g=-9.5,lowpass=f=14000"


def to24(path):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", str(SR), "-ac", "1", t], check=True)
    y, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    return y[:int(DUR * SR)]


def bands_arr(y):
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    f = librosa.fft_frequencies(sr=SR, n_fft=2048)
    tot = S.sum() + 1e-12
    return np.array([float(S[(f >= lo) & (f < hi)].sum() / tot)
                     for lo, hi in BANDS])


def analyze(path):
    y = to24(path)
    b = bands_arr(y)
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "16000", "-ac", "1", t], check=True)
    y16, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    fl = float(librosa.feature.spectral_flatness(
        y=y16[:60 * 16000], hop_length=256)[0].mean())
    return b, fl


if __name__ == "__main__":
    tgt = np.mean([bands_arr(to24(p)) for p in ELS], axis=0)
    b0, f0 = analyze(SRC)
    print("EL 目标\t" + "\t".join("%.4f" % x for x in tgt))
    print("裸片\t" + "\t".join("%.4f" % x for x in b0)
          + "\tL1=%.4f\tflat=%.5f" % (np.abs(b0 - tgt).sum(), f0))
    rows = []
    for F, W, G in itertools.product([80, 95, 110], [0.5, 0.7],
                                     [0, 4, 7, 10, 14]):
        af = FIX if not G else ("equalizer=f=%d:t=q:w=%g:g=%d," % (F, W, G)) + FIX
        o = tempfile.mktemp(suffix=".wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", SRC,
                        "-af", af, o], check=True)
        b, fl = analyze(o)
        os.unlink(o)
        rows.append((float(np.abs(b - tgt).sum()), F, W, G, fl, b, af))
    rows.sort(key=lambda r: r[0])
    print("\nL1\t中心\t宽度\t增益\tflat\t" + "\t".join(
        "%d-%d" % x for x in BANDS))
    for r in rows[:12]:
        print("%.4f\t%d\t%.1f\t+%d\t%.5f\t" % (r[0], r[1], r[2], r[3], r[4])
              + "\t".join("%.4f" % x for x in r[5]))
    print("\nBEST:", rows[0][6])
