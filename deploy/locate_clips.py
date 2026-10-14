#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""locate_clips.py — 用包络相关反查 h6 片段在母带中的位置"""
import numpy as np, soundfile as sf, subprocess, tempfile, os

BASE = "/mnt/tts/diag/cross-20261013"

def load16(path):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg","-y","-v","error","-i",path,"-ar","16000","-ac","1",t],check=True)
    y, sr = sf.read(t, dtype="float32"); os.unlink(t)
    return y, sr

def env(y, sr):
    fl = int(0.01*sr); n = len(y)//fl
    return np.sqrt((y[:n*fl].reshape(n,fl)**2).mean(axis=1))

def locate(clip_path, src_path, label):
    c, sr = load16(clip_path)
    s, _ = load16(src_path)
    ec, es = env(c, sr), env(s, sr)
    # 用 clip 前 20s 的包络做模板
    L = min(len(ec), 2000)
    tpl = ec[:L]
    tpl = tpl - tpl.mean()
    best, bo = -2, None
    # 全文扫描（步 0.2s）
    for off in range(0, len(es)-L, 20):
        seg = es[off:off+L]
        seg = seg - seg.mean()
        denom = np.linalg.norm(tpl)*np.linalg.norm(seg)
        if denom < 1e-9: continue
        cc = float((tpl*seg).sum()/denom)
        if cc > best:
            best, bo = cc, off
    t0 = bo*0.01
    dur_c = len(c)/sr
    print(f"{label:10s} {os.path.basename(clip_path):16s} <- {os.path.basename(src_path):14s} "
          f"位置={t0:7.2f}s 相关={best:.3f} 片段长={dur_c:.1f}s "
          f"占比={t0/(len(s)/sr)*100:.1f}%")
    return t0

locate(f"{BASE}/bank5/h6_fa_m.mp3", f"{BASE}/X2f_v2.wav", "F_mid")
locate(f"{BASE}/bank5/h6_fa_t.mp3", f"{BASE}/X2f_v2.wav", "F_tail")
locate(f"{BASE}/bank5/h6_fb_m.mp3", f"{BASE}/X4f_v2.wav", "Fb_mid")
locate(f"{BASE}/bank5/h6_ma_t.mp3", f"{BASE}/X1m_v3.wav", "M_tail")
