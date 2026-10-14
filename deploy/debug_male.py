#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""debug_male.py — 男声 v5 幸存脏停顿解剖 + silencedetect 阈值行为"""
import numpy as np, soundfile as sf, subprocess, re

BASE = "/mnt/tts/diag/cross-20261013"

def sd(path, thr, mind):
    r = subprocess.run(["ffmpeg","-nostats","-i",path,
        "-af",f"silencedetect=noise={thr}dB:d={mind}","-f","null","-"],
        capture_output=True,text=True)
    s=[float(x) for x in re.findall(r"silence_start: ([\d.]+)",r.stderr)]
    e=[float(x) for x in re.findall(r"silence_end: ([\d.]+)",r.stderr)]
    return list(zip(s,e))

def profile(path, t0, t1, tag):
    data, fs = sf.read(path)
    if data.ndim>1: data=data.mean(axis=1)
    i0,i1=int(t0*fs),int(t1*fs)
    seg=data[i0:i1]
    fl=int(0.01*fs); n=len(seg)//fl
    fr=seg[:n*fl].reshape(n,fl)
    db=20*np.log10(np.sqrt((fr**2).mean(axis=1))+1e-12)
    print(f"--- {tag} [{t0}-{t1}s] ---")
    for i in range(n):
        t=t0+i*0.01
        bar="#"*max(0,int((db[i]+90)/2.5))
        print(f"  {t:6.2f} {db[i]:7.1f} {bar}")

p5 = f"{BASE}/X1m_v5_v5.wav"
pr = f"{BASE}/X1_m.wav"

print("=== M_v5 在 14.4-15.0s（幸存脏点 14.61）===")
profile(p5, 14.4, 15.0, "M_v5")
print()
print("=== M_raw 在 18.8-19.4s（对应幸存点 19.01）===")
profile(pr, 18.8, 19.4, "M_raw")
print()
print("=== M_v5 在 18.8-19.4s ===")
profile(p5, 18.8, 19.4, "M_v5")
print()
print("=== silencedetect 阈值行为（M_v5 14.2-15.2 区间）===")
for thr in (-40, -45, -50):
    for a,b in sd(p5, thr, 0.15):
        if 14.0 < a < 15.5:
            print(f"  thr={thr}: gap [{a:.3f}, {b:.3f}] dur={b-a:.3f}")
print()
print("=== M_v5 19.0 附近各阈值 ===")
for thr in (-40, -45, -50):
    for a,b in sd(p5, thr, 0.15):
        if 18.5 < a < 19.8:
            print(f"  thr={thr}: gap [{a:.3f}, {b:.3f}] dur={b-a:.3f}")
