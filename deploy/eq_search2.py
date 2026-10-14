#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""「感冒音」专项 EQ 网格搜索（第二轮）。

背景：员工第 13 轮自由文本——「声音像感冒后发出的声音」。
band_fine.py 细分 7 段定位到：
    60-150Hz（胸腔共鸣）EL 0.0157 / 我们变调后 0.0067  —— 少 57%
    500-1kHz（鼻音共振区）EL 0.1649 / 我们变调后 0.2176 —— 多 32%
这正是"鼻音重 + 没底气"的感冒音特征。

根因不是 EQ，是**变调本身**：asetrate 把整个频谱（含共振峰）抬 1.213 倍，
    未变调 nof0 的 500-1k 是 0.1600（≈EL），变调后 0.2176；
    60-150 则相反，0.0322 → 0.0067（低频裙边被搬出该频段）。
所以修法只能是：变调**之后**再补一段校正 EQ。

搜索空间（都在变调后的 m_f0.wav 上叠）：
    低频补回    lowshelf=f=150:g=+X      X ∈ {0,2,3,4,5,6}
    鼻音区衰减  equalizer=f=F:q=Q:g=-Y   F ∈ {600,700,800}, Q ∈ {0.7,1.2}, Y ∈ {1,2,3}
    保留既有    highshelf=f=1500:g=1.5, highshelf=f=4000:g=-6.5, lowpass=f=14000
指标：7 段能量占比的 L1 偏差（对两条 EL 男声案例均值）+ flat 光谱平坦度（防过度加工）
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

TAIL = "highshelf=f=1500:g=1.5,highshelf=f=4000:g=-6.5,lowpass=f=14000"


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


def flat_of(y16):
    return float(librosa.feature.spectral_flatness(y=y16, hop_length=256)[0].mean())


def analyze(path):
    y = to24(path)
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "16000", "-ac", "1", t], check=True)
    y16, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    return bands_arr(y), flat_of(y16[:60 * 16000])


if __name__ == "__main__":
    tgt = np.mean([bands_arr(to24(p)) for p in ELS], axis=0)
    print("EL 目标\t" + "\t".join("%.4f" % x for x in tgt))
    b0, f0 = analyze(SRC)
    print("起点\t" + "\t".join("%.4f" % x for x in b0)
          + "\tL1=%.4f\tflat=%.5f" % (np.abs(b0 - tgt).sum(), f0))

    rows = []
    for X, F, Q, Y in itertools.product([0, 2, 3, 4, 5, 6],
                                        [600, 700, 800],
                                        [0.7, 1.2],
                                        [0, 1, 2, 3]):
        parts = []
        if X:
            parts.append("lowshelf=f=150:g=%d" % X)
        if Y:
            parts.append("equalizer=f=%d:t=q:w=%g:g=%d" % (F, Q, -Y))
        parts.append(TAIL)
        af = ",".join(parts)
        o = tempfile.mktemp(suffix=".wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", SRC,
                        "-af", af, o], check=True)
        b, fl = analyze(o)
        os.unlink(o)
        rows.append((float(np.abs(b - tgt).sum()), X, F, Q, Y, fl, b, af))

    rows.sort(key=lambda r: r[0])
    print("\nL1\t低补\t鼻F\t鼻Q\t鼻衰\tflat\t60-150\t500-1k")
    for r in rows[:20]:
        print("%.4f\t+%d\t%d\t%.1f\t-%d\t%.5f\t%.4f\t%.4f"
              % (r[0], r[1], r[2], r[3], r[4], r[5], r[6][0], r[6][3]))
    print("\nBEST:", rows[0][7])
