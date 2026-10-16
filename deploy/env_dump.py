#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把员工点名三处的 5ms 包络逐帧打出来（修前 vs 修后），看坑的具体形状。"""
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

TMP = "/mnt/tts/diag/_silfix"
SR = 24000
HOP = 0.005
SRC = "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"
PAIRS = [("现役", SRC), ("填充", os.path.join(TMP, "s_rtone_60_52.mp3"))]
MARKS = [(95.5, 94.6, 96.6), (131.3, 130.4, 132.4), (170.0, 169.1, 171.1)]


def load(src, tag):
    o = os.path.join(TMP, "_e%s.wav" % tag)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ac", "1",
                    "-ar", str(SR), o], check=True)
    return sf.read(o, dtype="float32")[0]


def env(y):
    h = int(HOP * SR)
    n = len(y) // h
    e = np.sqrt(np.array([(y[i * h:(i + 1) * h] ** 2).mean() for i in range(n)]))
    return 20 * np.log10(e + 1e-12)


def main():
    for t0, a, b in MARKS:
        print("\n########## 点名 %.1fs ##########" % t0)
        for name, p in PAIRS:
            db = env(load(p, "%s%d" % (name, int(t0 * 10))))
            i0, i1 = int(a / HOP), int(b / HOP)
            seg = db[i0:i1]
            print("-- %s（窗口最深 %6.1fdB @ %.2fs）--"
                  % (name, seg.min(), a + seg.argmin() * HOP))
            line = []
            for k in range(0, len(seg), 2):          # 每 10ms 一个点
                line.append("%.1f:%5.0f" % (a + k * HOP, seg[k]))
            for j in range(0, len(line), 8):
                print("   " + " ".join(line[j:j + 8]))


if __name__ == "__main__":
    main()
