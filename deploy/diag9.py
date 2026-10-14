#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diag9.py — A. 号码内部逐位结构（我们 vs EL）B. 男声残留定位"""
import numpy as np, soundfile as sf, subprocess, tempfile, os, re

BASE = "/mnt/tts/diag/cross-20261013"

def cut16(path, ss, to):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg","-y","-v","error","-ss",str(ss),"-to",str(to),
                    "-i",path,"-ar","16000","-ac","1",t], check=True)
    y, sr = sf.read(t, dtype="float32"); os.unlink(t)
    return y, sr

def profile(y, sr, thr_db=-38, min_sil=0.03):
    fl = int(0.01*sr); n = len(y)//fl
    fr = y[:n*fl].reshape(n, fl)
    db = 20*np.log10(np.sqrt((fr**2).mean(axis=1))+1e-12)
    voiced = db > thr_db
    rg = []
    i = 0
    while i < n:
        j = i
        while j+1 < n and voiced[j+1] == voiced[i]: j += 1
        t = "V" if voiced[i] else "G"
        if not (t=="G" and (j+1-i)*0.01 < min_sil):
            rg.append((t, round(i*0.01,3), round((j+1)*0.01,3), round(db[i:j+1].max(),1)))
        i = j+1
    return rg

print("=== A1. 我们的号码段 X2f_v3f [235.8~238.6] ===")
y, sr = cut16(f"{BASE}/X2f_v3f.wav", 235.8, 238.6)
for t,a,b,d in profile(y, sr):
    print(f"   {'浊音' if t=='V' else '间隙'} {a+235.8:7.2f}-{b+235.8:7.2f} {b-a:.3f}s peak={d}dB")
print("=== A2. EL 的号码段 5272 [228.2~231.0] ===")
y2, sr = cut16("/mnt/tts/data/cases/52727ec43154/merged.wav", 228.2, 231.0)
for t,a,b,d in profile(y2, sr):
    print(f"   {'浊音' if t=='V' else '间隙'} {a+228.2:7.2f}-{b+228.2:7.2f} {b-a:.3f}s peak={d}dB")

print()
print("=== B. 男声残留：检查 M_v5f 剩余脏停顿的帧剖面 ===")
data, fs = sf.read(f"{BASE}/X1m_v5f.wav")
if data.ndim > 1: data = data.mean(axis=1)
fl = int(0.01*fs)
for t0 in (29.99, 35.47, 196.2):
    i0, i1 = int((t0-0.05)*fs), int((t0+0.35)*fs)
    seg = data[i0:i1]
    n = len(seg)//fl
    fr = seg[:n*fl].reshape(n, fl)
    db = 20*np.log10(np.sqrt((fr**2).mean(axis=1))+1e-12)
    print(f"  t={t0}s:")
    print("   ", " ".join(f"{t0-0.05+i*0.01:.2f}:{v:.0f}" for i,v in enumerate(db[:22])))
