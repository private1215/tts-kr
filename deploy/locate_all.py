#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""locate_all.py — 对每个 h6 片段，试所有候选母带，报最佳匹配"""
import numpy as np, soundfile as sf, subprocess, tempfile, os, glob

BASE = "/mnt/tts/diag/cross-20261013"
CANDS = sorted(glob.glob(f"{BASE}/*.wav") + glob.glob("/mnt/tts/diag/campplus_rank-20261013/*.wav"))

def load16(path):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg","-y","-v","error","-i",path,"-ar","16000","-ac","1",t],check=True)
    y, sr = sf.read(t, dtype="float32"); os.unlink(t)
    return y, sr

def env(y, sr):
    fl = int(0.01*sr); n = len(y)//fl
    return np.sqrt((y[:n*fl].reshape(n,fl)**2).mean(axis=1))

clips = ["h6_fa_m","h6_fa_t","h6_ma_t","h6_sa_t","h6_fc_m"]
# 预加载母带包络
srcs = []
for p in CANDS:
    try:
        y, sr = load16(p)
        srcs.append((os.path.basename(p), env(y, sr), len(y)/sr))
        print("loaded", os.path.basename(p), flush=True)
    except Exception as e:
        print("skip", p, e)

for cname in clips:
    c, sr = load16(f"{BASE}/bank5/{cname}.mp3")
    ec = env(c, sr)
    L = min(len(ec), 2000)
    tpl = ec[:L]; tpl = tpl - tpl.mean()
    tn = np.linalg.norm(tpl)
    results = []
    for name, es, dur in srcs:
        if len(es) < L: continue
        best, bo = -2, None
        for off in range(0, len(es)-L, 40):  # 0.4s 步
            seg = es[off:off+L]; seg = seg - seg.mean()
            d = tn*np.linalg.norm(seg)
            if d < 1e-9: continue
            cc = float((tpl*seg).sum()/d)
            if cc > best: best, bo = cc, off
        results.append((best, name, bo*0.01, dur))
    results.sort(reverse=True)
    print(f"\n=== {cname} (时长{len(c)/sr:.1f}s) 最佳匹配 top3 ===")
    for cc, name, t0, dur in results[:3]:
        print(f"  {cc:.3f} {name} @ {t0:.1f}s (占比{t0/dur*100:.1f}%)")
