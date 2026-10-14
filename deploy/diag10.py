#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diag10.py — 号码逐位音节定位（浊音峰），给出分组边界插入点"""
import numpy as np, soundfile as sf, subprocess, tempfile, os

BASE = "/mnt/tts/diag/cross-20261013"
OURS, ELM = f"{BASE}/X2f_v3f.wav", "/mnt/tts/data/cases/52727ec43154/merged.wav"

def cut16(path, ss, to):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg","-y","-v","error","-ss",str(ss),"-to",str(to),
                    "-i",path,"-ar","16000","-ac","1",t], check=True)
    y, sr = sf.read(t, dtype="float32"); os.unlink(t)
    return y, sr

def syl_peaks(y, sr, base_t, thr_rel=0.12):
    """10ms 包络峰检测：返回 (时间, 相对电平) 列表"""
    fl = int(0.01*sr); n = len(y)//fl
    fr = y[:n*fl].reshape(n, fl)
    env = np.sqrt((fr**2).mean(axis=1))
    env = np.convolve(env, np.ones(3)/3, mode="same")
    mx = np.percentile(env, 85)
    peaks = []
    for i in range(1, n-1):
        if env[i] > mx*thr_rel and env[i] >= env[i-1] and env[i] > env[i+1]:
            # 相邻峰最小间隔 80ms
            if peaks and base_t + i*0.01 - peaks[-1][0] < 0.08:
                if env[i] > peaks[-1][1]:
                    peaks[-1] = (base_t + i*0.01, env[i])
                continue
            peaks.append((base_t + i*0.01, float(env[i])))
    return peaks

for name, path, ss, to in [("OURS", OURS, 235.6, 238.6), ("EL", ELM, 228.0, 231.2)]:
    y, sr = cut16(path, ss, to)
    pk = syl_peaks(y, sr, ss)
    print(f"\n=== {name} [{ss}~{to}s] 音节峰 {len(pk)} 个 ===")
    for i,(t,a) in enumerate(pk):
        print(f"   #{i+1:2d} t={t:7.2f}s amp={a:.4f}")
