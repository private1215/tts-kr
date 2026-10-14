#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diag11.py — 号码后段电平塌陷溯源：raw / 拉伸态 / 清洗后 / EL 对比"""
import numpy as np, soundfile as sf, subprocess, tempfile, os

BASE = "/mnt/tts/diag/cross-20261013"

def cut16(path, ss, to):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg","-y","-v","error","-ss",str(ss),"-to",str(to),
                    "-i",path,"-ar","16000","-ac","1",t], check=True)
    y, sr = sf.read(t, dtype="float32"); os.unlink(t)
    return y, sr

def block_db(y, sr, base, blk=0.25):
    """每 0.25s 一块，报块内峰值 dB 与 RMS dB"""
    fl = int(blk*sr); n = len(y)//fl
    fr = y[:n*fl].reshape(n, fl)
    rms = 20*np.log10(np.sqrt((fr**2).mean(axis=1))+1e-12)
    pk = 20*np.log10(np.abs(fr).max(axis=1)+1e-12)
    return [(round(base+i*blk,2), round(r,1), round(p,1)) for i,(r,p) in enumerate(zip(rms,pk))]

for name, path, ss, to in [
    ("RAW  X2_f.wav   (TTS裸输出)", f"{BASE}/X2_f.wav",   228.0, 242.0),
    ("ST   X2f_st.wav (仅atempo)",   f"{BASE}/X2f_st.wav", 228.0, 242.0),
    ("V3   X2f_v3f    (清洗后)",     f"{BASE}/X2f_v3f.wav",228.0, 242.0),
    ("EL   5272",                    "/mnt/tts/data/cases/52727ec43154/merged.wav", 224.0, 238.0),
]:
    try:
        y, sr = cut16(path, ss, to)
    except Exception as e:
        print(f"{name}: 跳过({e})"); continue
    print(f"\n=== {name} [{ss}~{to}s] ===")
    print("   起(s)   RMSdB  峰值dB")
    for t, r, p in block_db(y, sr, ss):
        bar = "#"*max(0, int((r+70)/3))
        print(f"   {t:7.2f} {r:7.1f} {p:7.1f} {bar}")
