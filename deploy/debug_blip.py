#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""debug_blip.py — v4 极端凸起的形态解剖（时长/频谱/孤立性）"""
import numpy as np, soundfile as sf

BASE = "/mnt/tts/diag/cross-20261013"
path = f"{BASE}/X1m_v3.wav"  # v4 内容
data, fs = sf.read(path)
if data.ndim > 1: data = data.mean(axis=1)
fl = int(0.01 * fs)

def frames_db(t0, t1):
    t0 = max(0.0, t0)
    i0, i1 = int(t0 * fs), int(t1 * fs)
    seg = data[i0:i1]
    n = len(seg) // fl
    fr = seg[:n * fl].reshape(n, fl)
    return 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-12), fr

spots = [0.09, 43.49, 202.08, 224.16, 395.85, 439.99]
for t in spots:
    db, fr = frames_db(t - 0.35, t + 0.35)
    pk = int(np.argmax(db))
    pkt = t - 0.35 + pk * 0.01
    # 凸起持续时间（>峰值-12dB 的连续帧）
    thr = db[pk] - 12
    lo = pk
    while lo > 0 and db[lo - 1] > thr: lo -= 1
    hi = pk
    while hi < len(db) - 1 and db[hi + 1] > thr: hi += 1
    dur = (hi - lo + 1) * 0.01
    # 凸起中段频谱
    c0 = (lo + (hi - lo) // 2) * fl
    bump = fr[(lo + (hi - lo) // 2)]
    w = np.hanning(fl)
    sp = np.abs(np.fft.rfft(bump * w)) ** 2
    freqs = np.fft.rfftfreq(fl, 1 / fs)
    top = np.argsort(sp)[-6:][::-1]
    # 孤立性：凸起前后 0.1s 的电平
    pre = db[max(0, lo - 10):lo].mean() if lo >= 10 else float("nan")
    post = db[hi + 1:hi + 11].mean() if hi + 11 <= len(db) else float("nan")
    print(f"t={t:7.2f} 峰值帧t={pkt:7.2f} 峰值={db[pk]:6.1f}dB 持续={dur*1000:5.0f}ms "
          f"前{pre:6.1f}dB 后{post:6.1f}dB")
    print(f"         主频: {[(round(freqs[i]), round(10*np.log10(sp[i]+1e-12),1)) for i in top]}")
