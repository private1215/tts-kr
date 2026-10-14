#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F0 沿时间轴的漂移曲线（男声）。

🔴 怀疑：VoxCPM2 的 F0 在长篇里会**往上漂**，而 EL 是稳的。
   实测 raw 前60s 169.7 / 全篇 185.1（+9%），EL 案例 196.1/194.9（−0.6%）。
   fix_chain 现在按「前 60s」定一个**全局**变调比例，于是：
     开头 163 → 197.7（对）  但后半段 185×1.213 ≈ 224（比 EL 高 13%，发尖发假）
   如果漂移坐实，正确修法不是整体变调，而是**分窗变调**。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fix_chain import f0_median, load16  # noqa: E402

SR = 16000
WIN = 60.0

FILES = [
    ("VoxCPM2 raw", "/mnt/tts/diag/x8197/vox_male_raw.wav"),
    ("EL 8197", "/mnt/tts/data/cases/8197d9ac8a35/merged.wav"),
    ("EL 1a6c", "/mnt/tts/data/cases/1a6c20a60481/merged.wav"),
    ("v6 全局变调(旧)", "/mnt/tts/diag/x8197/base_p.wav"),
    ("v7 分窗变调(新)", "/mnt/tts/diag/x8197/base_pw.wav"),
]


def profile(path):
    y, sr = load16(path)
    n = int(WIN * sr)
    out = []
    i = 0
    while i + int(20 * sr) < len(y):
        seg = y[i:i + n]
        if len(seg) < int(20 * sr):
            break
        f = f0_median(seg, sr, win=1e9)
        out.append(f if f else float("nan"))
        i += n
    return out


if __name__ == "__main__":
    print("文件\t" + "\t".join("%.0f-%ds" % (i * WIN, (i + 1) * WIN)
                               for i in range(9)))
    for name, p in FILES:
        if not os.path.exists(p):
            print(name, "缺失")
            continue
        v = profile(p)
        print("%-22s" % name + "\t".join("%.1f" % x for x in v)
              + "\t| 极差 %.1f Hz" % (np.nanmax(v) - np.nanmin(v)))
