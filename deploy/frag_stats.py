#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""frag_stats.py — 填充词（谐波孤立段）的电平/时长分布对比：ours vs EL"""
import numpy as np, soundfile as sf, sys
sys.path.insert(0, "/mnt/tts/diag/cross-20261013")

def scan(path):
    data, fs = sf.read(path)
    if data.ndim > 1: data = data.mean(axis=1)
    fl = int(0.01 * fs)
    n = len(data) // fl
    fr = data[:n * fl].reshape(n, fl)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-12)
    half = 50
    floor = np.array([np.percentile(db[max(0, i - half):min(n, i + half + 1)], 20) for i in range(n)])
    loud = db > np.maximum(floor + 12.0, -48.0)
    runs = []
    i = 0
    while i < n:
        if loud[i]:
            j = i
            while j + 1 < n and (loud[j + 1] or (j + 2 < n and loud[j + 2])):
                j += 1
            runs.append((i, j)); i = j + 1
        else:
            i += 1
    out = []
    for (a, b) in runs:
        durf = b - a + 1
        if durf < 2 or durf > 16: continue
        pre = db[max(0, a - 8):a]; post = db[b + 1:min(n, b + 9)]
        if not ((len(pre) == 0 or np.median(pre) < -45) and (len(post) == 0 or np.median(post) < -45)):
            continue
        seg = data[a * fl:(b + 1) * fl]
        x = seg - seg.mean()
        if np.sqrt((x ** 2).mean()) < 1e-6: continue
        ac = np.correlate(x, x, mode="full")[len(x) - 1:]
        ac /= (ac[0] + 1e-12)
        h = float(ac[int(0.0025 * fs):int(0.0125 * fs)].max())
        if h <= 0.45: continue
        lvl = float(db[a:b + 1].max())
        out.append((a * 0.01, durf * 10, lvl))
    return out

BASE = "/mnt/tts/diag/cross-20261013"
for label, path in [
    ("EL 女5272", "/mnt/tts/data/cases/52727ec43154/merged.wav"),
    ("F_v3 女", f"{BASE}/X2f_v3.wav"),
    ("EL 男8197", "/mnt/tts/data/cases/8197d9ac8a35/merged.wav"),
    ("M_v5 男", f"{BASE}/X1m_v5_v5.wav"),
    ("M_v4 男(员工听的)", f"{BASE}/X1m_v3.wav"),
]:
    frs = scan(path)
    if not frs:
        print(f"{label:20s} 无"); continue
    lv = np.array([f[2] for f in frs]); du = np.array([f[1] for f in frs])
    early = [f for f in frs if f[0] < 2.0]
    print(f"{label:20s} n={len(frs):3d} 时长中位={np.median(du):5.0f}ms "
          f"电平: p25={np.percentile(lv,25):6.1f} 中位={np.median(lv):6.1f} p75={np.percentile(lv,75):6.1f} "
          f"前2秒内={len(early)}")
