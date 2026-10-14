#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""post_scrub.py — 对成品 wav 跑孤立人声碎片清除（scrub_stray），
报告清除处数，并输出清洗后文件。用法: post_scrub.py <in.wav> <out.wav>"""
import sys, numpy as np, soundfile as sf
sys.path.insert(0, "/mnt/tts/diag/cross-20261013")
from pause_clean_v3 import scrub_stray

inp, outp = sys.argv[1], sys.argv[2]
data, fs = sf.read(inp)
mono = data.ndim == 1
y = data if mono else data.mean(axis=1)
y2, n, hits = scrub_stray(y.copy(), fs)
print(f"{inp}: 清除孤立碎片 {n} 处")
for h in hits:
    print(f"   t={h[0]}s 时长{h[1]:.0f}ms")
if mono:
    sf.write(outp, y2, fs)
else:
    # 多声道：对所有声道应用同一替换（本流程全是单声道，防御性保留）
    out = np.stack([scrub_stray(data[:, c].copy(), fs)[0] for c in range(data.shape[1])], axis=1)
    sf.write(outp, out, fs)
print("saved:", outp)
