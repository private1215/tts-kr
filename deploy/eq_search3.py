#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""「感冒音」EQ 网格搜索（第三轮，精细化）。

第二轮结论（eq_search2.py）：
  · 鼻音区衰减是主力：equalizer=f=800:w=1.2:g=-2，L1 0.1569 → 0.0975
  · ❌ lowshelf=f=150 补低频**反而变差**（0.0975 → 0.1150）：
    ffmpeg lowshelf 默认 slope=0.5，过渡带极宽，连带把 150~500Hz 也抬了，
    而这两段我们已经不缺（甚至偏多）。要补 60-150 必须用**窄带 peaking**。

本轮改为「全 peaking」结构，独立控制每一段：
  低补   equalizer=f=100:w=1.0:g=+A      （只覆盖 ~70-140Hz）
  鼻音   equalizer=f=800:w=1.2:g=-B
  存在感 highshelf=f=1500:g=+C           （补 1k-2k）
  亮薄   highshelf=f=4000:g=-D           （压变调搬上来的 2k-8k）
指标：7 段 L1（对两条 EL 男声案例均值）+ flat（EL 实测 0.042，防过度加工）
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

SRC = os.environ.get("SRC", "/mnt/tts/diag/x8197/m_f0.wav")
ELS = ["/mnt/tts/data/cases/8197d9ac8a35/merged.wav",
       "/mnt/tts/data/cases/1a6c20a60481/merged.wav"]


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
    print("EL 目标\t" + "\t".join("%.4f" % x for x in tgt) + "\t\tflat")
    print("起点\t" + "\t".join("%.4f" % x for x in b0)
          + "\tL1=%.4f\tflat=%.5f" % (np.abs(b0 - tgt).sum(), f0))

    rows = []
    for A, B, C, D in itertools.product([0, 3, 5, 7],
                                        [1, 2, 3],
                                        [1.5, 2.5, 3.5],
                                        [6.5, 8.0, 9.5]):
        parts = []
        if A:
            parts.append("equalizer=f=100:t=q:w=1.0:g=%d" % A)
        parts.append("equalizer=f=800:t=q:w=1.2:g=%d" % -B)
        parts.append("highshelf=f=1500:g=%g" % C)
        parts.append("highshelf=f=4000:g=%g" % -D)
        parts.append("lowpass=f=14000")
        af = ",".join(parts)
        o = tempfile.mktemp(suffix=".wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", SRC,
                        "-af", af, o], check=True)
        b, fl = analyze(o)
        os.unlink(o)
        rows.append((float(np.abs(b - tgt).sum()), A, B, C, D, fl, b, af))

    rows.sort(key=lambda r: r[0])
    print("\nL1\t低补\t鼻衰\t1.5k\t4k\tflat\t" + "\t".join(
        "%d-%d" % x for x in BANDS))
    for r in rows[:15]:
        print("%.4f\t+%d\t-%d\t+%.1f\t-%.1f\t%.5f\t" % (r[0], r[1], r[2],
                                                        r[3], r[4], r[5])
              + "\t".join("%.4f" % x for x in r[6]))
    print("\nBEST:", rows[0][7])
