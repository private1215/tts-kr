#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify2.py — 修正版停顿验收：只分析每处停顿的中段（边缘各砍 max(30ms, 25%)），
排除语音起音/落音沿的边界帧（自然过渡，非脏）。
脏 = 中段内帧 RMS 最大-中位 > 8dB。"""
import numpy as np, soundfile as sf, subprocess, re, sys

def detect_gaps_wav(path, thr=-45, mind=0.15):
    r = subprocess.run(["ffmpeg", "-nostats", "-i", path,
                        "-af", f"silencedetect=noise={thr}dB:d={mind}", "-f", "null", "-"],
                       capture_output=True, text=True)
    s = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r.stderr)]
    e = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    return list(zip(s, e))

def verify(path, label):
    data, fs = sf.read(path)
    if data.ndim > 1: data = data.mean(axis=1)
    gaps = detect_gaps_wav(path)
    dirty, details = 0, []
    floors = []
    for a, b in gaps:
        d = b - a
        trim = max(0.03, d * 0.25)
        i0, i1 = int((a + trim) * fs), int((b - trim) * fs)
        seg = data[i0:i1]
        fl = int(0.01 * fs)
        n = len(seg) // fl
        if n < 3:
            continue
        fr = seg[:n * fl].reshape(n, fl)
        rms = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-12)
        med = float(np.median(rms))
        above = float(rms.max()) - med
        floors.append(med)
        if above > 8:
            dirty += 1
            details.append((round(a, 2), round(d, 2), round(above, 1)))
    fm = float(np.median(floors)) if floors else float("nan")
    print(f"{label:28s} 停顿={len(gaps):3d} 中段脏={dirty:3d} 底噪中位={fm:6.1f}dB")
    for x in details[:10]:
        print(f"     脏: t={x[0]} dur={x[1]} 凸起={x[2]}dB")
    return len(gaps), dirty

BASE = "/mnt/tts/diag/cross-20261013"
print("=== 女声 ===")
verify(f"{BASE}/X2_f.wav",    "F_raw  OmniVoice裸推理")
verify(f"{BASE}/X2f_fix.wav", "F_v1  复制每处底噪版")
verify(f"{BASE}/X2f_v2.wav",  "F_v2  RMS最低模板版(员工听的)")
verify(f"{BASE}/X2f_v3.wav",  "F_v3  合成tone版(新)")
print("=== 男声 ===")
verify(f"{BASE}/X1_m.wav",    "M_raw VoxCPM2裸推理")
verify(f"{BASE}/X1m_fix.wav", "M_v2  模板插入版")
verify(f"{BASE}/X1m_v3.wav",  "M_v4  atempo+EQ(员工听的)")
verify(f"{BASE}/X1m_v5_v5.wav", "M_v5  v3语速+合成tone清洗(新)")
print("=== EL 标杆（同口径参照）===")
verify("/mnt/tts/data/cases/52727ec43154/merged.wav", "EL 女5272")
verify("/mnt/tts/data/cases/8197d9ac8a35/merged.wav", "EL 男8197")
