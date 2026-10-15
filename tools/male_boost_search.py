#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""男声 150~300Hz 衰减方案对比（针对「感冒闷」的客观对应项）。

现役男声成品实测 150~300 = +3.96σ，与员工「像感冒鼻塞」精确对应。
自动搜索器选的是「2000Hz +2dB」（提升高频来相对降低低频占比），
副作用是把 1k~2k 推到 +2.73σ。这里改试**直接衰减**低频区，副作用应该更小。
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_g import measure, load_base  # noqa: E402

SRC = "/mnt/tts/diag/xf5272/prod_EL_male_8197d9ac8a35.wav"
base = load_base("male")
USE = ["b60_150", "b150_300", "b300_500", "b500_1000",
       "b1000_2000", "b2000_4000", "b4000_8000", "flat"]

CANDS = [
    ("(现状)", None),
    ("250Hz -0.5", "equalizer=f=250:t=q:w=1.1:g=-0.5"),
    ("250Hz -0.75", "equalizer=f=250:t=q:w=1.1:g=-0.75"),
    ("250Hz -1.0", "equalizer=f=250:t=q:w=1.1:g=-1"),
    ("250Hz -1.5", "equalizer=f=250:t=q:w=1.1:g=-1.5"),
    ("250Hz -2.0", "equalizer=f=250:t=q:w=1.1:g=-2"),
    ("350Hz -0.5", "equalizer=f=350:t=q:w=1.1:g=-0.5"),
    ("350Hz -1.0", "equalizer=f=350:t=q:w=1.1:g=-1"),
]

print("%-16s %8s | %8s %8s %8s %8s %8s"
      % ("方案", "Σz²", "150~300", "300~500", "1k~2k", "2k~4k", "flat"))
for i, (name, af) in enumerate(CANDS):
    if af is None:
        p = SRC
    else:
        p = "/mnt/tts/diag/_mt%d.wav" % i
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", SRC,
                        "-af", af, p], check=True)
    m = measure(p)
    z = {k: (m[k] - base[k]["mean"]) / base[k]["sd"] for k in USE}
    J = sum(v * v for v in z.values())
    print("%-16s %8.1f | %+8.2f %+8.2f %+8.2f %+8.2f %+8.2f"
          % (name, J, z["b150_300"], z["b300_500"],
             z["b1000_2000"], z["b2000_4000"], z["flat"]))
