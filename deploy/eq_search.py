#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""搜索一组 EQ，把我们成品的频段能量分布拉到 EL 的样子。

可靠测量（24k 口径，60s）：
    EL 案例8197 : 100-1k 0.712 / 1k-4k 0.275 / 4k-8k 0.0108
    我们成品 m_v4: 100-1k 0.763 / 1k-4k 0.225 / 4k-8k 0.0124
我们低频(100~1k)偏多 5 个点、中频(1~4k)偏少 5 个点 —— 听感上就是"厚而闷、不够清晰"。
"""
import subprocess
import tempfile
import os
import numpy as np
import soundfile as sf
import librosa

SRC = "/mnt/tts/diag/x8197/m_v4.wav"
TGT = {"100-1k": 0.712, "1k-4k": 0.275, "4k-8k": 0.0108}
SR = 24000
DUR = 60.0

CANDS = [
    ("当前(基线)", "anull"),
    ("A 低-0.5 中+1.0", "lowshelf=f=800:g=-0.5,highshelf=f=1500:g=1.0"),
    ("B 低-1.0 中+1.5", "lowshelf=f=800:g=-1.0,highshelf=f=1500:g=1.5"),
    ("C 低-1.5 中+2.0", "lowshelf=f=1000:g=-1.5,highshelf=f=2000:g=2.0"),
    ("D 低-1.0 中+2.5", "lowshelf=f=900:g=-1.0,highshelf=f=1800:g=2.5"),
    ("E 低-2.0 中+3.0", "lowshelf=f=900:g=-2.0,highshelf=f=1800:g=3.0"),
    ("F 低-1.0 中+1.5 高再压3", "lowshelf=f=800:g=-1.0,highshelf=f=1500:g=1.5,highshelf=f=4000:g=-3.0"),
    ("H 低-1.0 中+1.5 高压1.5", "lowshelf=f=800:g=-1.0,highshelf=f=1500:g=1.5,highshelf=f=4000:g=-1.5"),
    ("I 低-1.0 中+1.8 高压2.2", "lowshelf=f=800:g=-1.0,highshelf=f=1500:g=1.8,highshelf=f=4000:g=-2.2"),
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
    out = {}
    for lo, hi, name in [(100, 1000, "100-1k"), (1000, 4000, "1k-4k"),
                         (4000, 8000, "4k-8k")]:
        m = (f >= lo) & (f < hi)
        out[name] = S[m].sum() / tot
    return out


def flat(path):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "16000", "-ac", "1", t], check=True)
    y, sr = sf.read(t, dtype="float32")
    os.unlink(t)
    y = y[:int(60 * sr)]
    return float(librosa.feature.spectral_flatness(y=y, hop_length=256)[0].mean())


if __name__ == "__main__":
    el = bands("/mnt/tts/data/cases/8197d9ac8a35/merged.wav")
    print("EL 目标\t" + "\t".join("%.4f" % el[k] for k in TGT))
    print("候选\t100-1k\t1k-4k\t4k-8k\t偏差L1\tflat")
    for name, af in CANDS:
        o = tempfile.mktemp(suffix=".wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", SRC,
                        "-af", af, o], check=True)
        b = bands(o)
        err = sum(abs(b[k] - TGT[k]) for k in TGT)
        print("%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.5f"
              % (name, b["100-1k"], b["1k-4k"], b["4k-8k"], err, flat(o)))
        os.unlink(o)
