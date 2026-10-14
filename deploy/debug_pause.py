#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""debug_pause.py — 直接看 X2f_v3.wav 在 t=15.13 附近到底有什么"""
import numpy as np, soundfile as sf, subprocess, re, sys
sys.path.insert(0, "/mnt/tts/diag/cross-20261013")
from pause_clean_v3 import detect_gaps_wav, synth_tone

path = "/mnt/tts/diag/cross-20261013/X2f_v3.wav"
data, fs = sf.read(path)
if data.ndim > 1: data = data.mean(axis=1)
print("file:", path, "fs:", fs, "dur: %.2f" % (len(data)/fs))

# 该处附近的 silencedetect 间隙
gaps, dur = detect_gaps_wav(path, thr=-45, mind=0.15)
near = [(a, d) for a, d in gaps if 14.5 < a < 15.8]
print("15s附近的≥0.15s间隙:", near)

gaps8, _ = detect_gaps_wav(path, thr=-45, mind=0.03)
near8 = [(a, d) for a, d in gaps8 if 14.5 < a < 15.8]
print("15s附近的≥0.03s间隙:", near8)

# 帧 RMS 剖面 14.8-15.6
i0, i1 = int(14.8*fs), int(15.6*fs)
seg = data[i0:i1]
fl = int(0.01*fs)
n = len(seg)//fl
fr = seg[:n*fl].reshape(n, fl)
db = 20*np.log10(np.sqrt((fr**2).mean(axis=1))+1e-12)
print("帧RMS剖面(10ms) 14.80s起:")
for i in range(0, n, 2):
    t = 14.8 + i*0.01
    bar = "#" * max(0, int((db[i]+80)/2))
    print(f"  {t:6.2f} {db[i]:7.1f} {bar}")

# 最响帧的频谱
pk = int(np.argmax(db))
c0 = pk*fl; c1 = c0+fl
w = np.hanning(fl)
sp = np.abs(np.fft.rfft(seg[c0:c1]*w))**2
freqs = np.fft.rfftfreq(fl, 1/fs)
top = np.argsort(sp)[-10:][::-1]
print("最响帧 t=%.2f 主频:" % (14.8+pk*0.01),
      [(round(freqs[i]), round(10*np.log10(sp[i]+1e-12),1)) for i in top])

# 对照：合成 tone 自己的帧RMS波动
tone = synth_tone(int(0.18*fs), fs, 10**(-59.2/20))
n2 = len(tone)//fl
fr2 = tone[:n2*fl].reshape(n2, fl)
db2 = 20*np.log10(np.sqrt((fr2**2).mean(axis=1))+1e-12)
print("合成tone帧RMS: 中位=%.1f 最大=%.1f 凸起=%.1f dB" % (
    np.median(db2), db2.max(), db2.max()-np.median(db2)))
