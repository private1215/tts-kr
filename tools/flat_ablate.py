# -*- coding: utf-8 -*-
"""flat 消融：逐个 EQ 滤波器量频谱平坦度，定位到底是谁在杀 flat。

背景（2026-10-14）：v11 把所有频段都压进 1.5σ，但 flat 从 0.0493 掉到 0.0350
（-3.69σ），共振峰反而比 v10 更差。怀疑是某个大增益滤波器把频谱搞"尖"了
（flat = 几何均值/算术均值，局部猛抬 -> 算术均值涨得更快 -> flat 掉）。
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eq_search6 as E  # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else "base_pw.wav"
src = SRC if os.path.isabs(SRC) else os.path.join(HERE, SRC)

CASES = [
    ("不加 EQ", ""),
    ("lowpass 14000", "lowpass=f=14000"),
    ("90 +8", "equalizer=f=90:t=q:w=1.1:g=8"),
    ("250 +2", "equalizer=f=250:t=q:w=1.1:g=2"),
    ("450 +3", "equalizer=f=450:t=q:w=1.1:g=3"),
    ("1400 +4", "equalizer=f=1400:t=q:w=1.1:g=4"),
    ("highshelf4000 -1", "highshelf=f=4000:g=-1"),
    ("highshelf4000 -9.5", "highshelf=f=4000:g=-9.5"),
    ("v11 全套", E.eq_str([8, 2, 3, 0, 4, 0, 2, -2])),
]
V10 = ("equalizer=f=90:t=q:w=1.1:g=7.5,"
       "equalizer=f=350:t=q:w=1.1:g=3,"
       "equalizer=f=1400:t=q:w=1.1:g=3,"
       "highshelf=f=4000:g=-1,lowpass=f=14000")
CASES.append(("v10 全套", V10))

print("%-22s %8s %8s" % ("滤波器", "flat", "z"))
print("-" * 42)
for name, af in CASES:
    t = tempfile.mktemp(suffix=".wav")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", src, "-ar", "24000", "-ac", "1"]
    if af:
        cmd += ["-af", af]
    cmd.append(t)
    subprocess.run(cmd, check=True)
    y16, _ = E.load16(t)
    flat = float(np.mean(E.librosa.feature.spectral_flatness(
        y=y16[:60 * 16000], hop_length=256)[0]))
    os.unlink(t)
    mm, ss = E.BASE["flat"]
    print("%-22s %8.4f  %+6.2f" % (name, flat, (flat - mm) / ss))
print("\nEL 基线 flat = %.4f ± %.4f" % E.BASE["flat"])
