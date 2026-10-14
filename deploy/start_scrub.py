#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""start_scrub.py — 只清除文件开头 1.5s 内的孤立响段（生成开头伪影，
EL 无此特征）。不碰填充词/呼吸（那些 EL 也有，是自然特征）。
用法: start_scrub.py <in.wav> <out.wav>"""
import sys, numpy as np, soundfile as sf
sys.path.insert(0, "/mnt/tts/diag/cross-20261013")
from pause_clean_v3 import synth_tone

inp, outp = sys.argv[1], sys.argv[2]
data, fs = sf.read(inp)
y = data if data.ndim == 1 else data.mean(axis=1)
fl = int(0.01 * fs)
n = len(y) // fl
fr = y[:n * fl].reshape(n, fl)
db = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-12)
# 局部底噪（前 3s 的 20 分位）
zone = min(n, 300)
floor = np.percentile(db[:zone], 20)
loud = db > np.maximum(floor + 12.0, -48.0)
# 前 1.5s 内的连续响段
i = 0
hits = []
while i < min(n, 150):
    if loud[i]:
        j = i
        while j + 1 < n and (loud[j + 1] or (j + 2 < n and loud[j + 2])):
            j += 1
        durf = j - i + 1
        if 2 <= durf <= 16:
            post = db[j + 1:min(n, j + 9)]
            if len(post) == 0 or np.median(post) < -45:
                hits.append((i, j))
        i = j + 1
    else:
        i += 1
w = max(1, int(0.005 * fs))
ramp = np.linspace(0, 1, w)
for (a, b) in hits:
    i0 = max(0, a * fl - int(0.015 * fs))
    i1 = min(len(y), (b + 1) * fl + int(0.015 * fs))
    tone = synth_tone(i1 - i0, fs, 10 ** (floor / 20.0))
    ol, orr = y[i0:i0 + w].copy(), y[i1 - w:i1].copy()
    y[i0:i1] = tone
    y[i0:i0 + w] = ol * (1 - ramp) + tone[:w] * ramp
    y[i1 - w:i1] = orr * ramp[::-1] + tone[-w:] * ramp[::-1]
print(f"{inp}: 开头伪影清除 {len(hits)} 处",
      [(round(a * 0.01, 2), round((b - a + 1) * 10)) for a, b in hits])
sf.write(outp, y, fs)
print("saved:", outp)
