#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""classify_frag.py — 给孤立碎片分类：呼吸（宽带噪声）vs 人声碎片（谐波）。
谐波度 = 80-400Hz 滞后范围归一化自相关峰值。
对 EL 男/EL 女/M_v5/F_v3 四份文件分别统计。"""
import numpy as np, soundfile as sf, sys
sys.path.insert(0, "/mnt/tts/diag/cross-20261013")
from pause_clean_v3 import synth_tone

def frame_db(buf, sr):
    fl = int(0.01 * sr)
    n = len(buf) // fl
    fr = buf[:n * fl].reshape(n, fl)
    return 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-12), fl, n

def harmonicity(seg, sr):
    """归一化自相关峰值（滞后 40-200 样本 @16k = 250-400Hz）。"""
    x = seg - seg.mean()
    if len(x) < 256 or np.sqrt((x ** 2).mean()) < 1e-6:
        return 0.0
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac /= (ac[0] + 1e-12)
    lo, hi = int(0.0025 * sr), int(0.0125 * sr)  # 80-400Hz
    return float(ac[lo:hi].max()) if hi < len(ac) else 0.0

def scan(path, label):
    data, fs = sf.read(path)
    if data.ndim > 1: data = data.mean(axis=1)
    db, fl, n = frame_db(data, fs)
    half = int(0.5 * sr if False else 0.5 * fs / fl)
    floor = np.array([np.percentile(db[max(0, i - half):min(n, i + half + 1)], 20)
                      for i in range(n)])
    loud = db > np.maximum(floor + 12.0, -48.0)
    runs = []
    i = 0
    while i < n:
        if loud[i]:
            j = i
            while j + 1 < n and (loud[j + 1] or (j + 2 < n and loud[j + 2])):
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    ctx = 8
    voiced, noise = [], []
    for (a, b) in runs:
        durf = b - a + 1
        if durf < 2 or durf > 16:
            continue
        pre = db[max(0, a - ctx):a]
        post = db[b + 1:min(n, b + 1 + ctx)]
        if not ((len(pre) == 0 or np.median(pre) < -45) and
                (len(post) == 0 or np.median(post) < -45)):
            continue
        seg = data[a * fl:(b + 1) * fl]
        h = harmonicity(seg, fs)
        (voiced if h > 0.45 else noise).append((a * 0.01, durf * 10, round(h, 2)))
    print(f"{label:22s} 孤立段: 谐波(人声碎片)={len(voiced):3d}  宽带(呼吸?)={len(noise):3d}")
    for v in voiced[:12]:
        print(f"     VOICED t={v[0]:7.2f} {v[1]:3.0f}ms h={v[2]}")
    for x in noise[:5]:
        print(f"     breath t={x[0]:7.2f} {x[1]:3.0f}ms h={x[2]}")
    return voiced, noise

BASE = "/mnt/tts/diag/cross-20261013"
print("=== EL 标杆 ===")
scan("/mnt/tts/data/cases/52727ec43154/merged.wav", "EL 女5272")
scan("/mnt/tts/data/cases/8197d9ac8a35/merged.wav", "EL 男8197")
print("=== 我们的成品 ===")
scan(f"{BASE}/X2f_v3.wav", "F_v3 女声")
scan(f"{BASE}/X1m_v5_v5.wav", "M_v5 男声")
