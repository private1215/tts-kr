#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diag8.py — 第七轮两处残留的客观定位
A. 号码段：把我们的片段对齐到 EL，比逐位数字的「浊音段/间隙」结构
B. 男声残留：确认剩余脏点是起音伪影还是真脏 + 填充词位置特征 vs EL
"""
import numpy as np, soundfile as sf, subprocess, tempfile, os, re

BASE = "/mnt/tts/diag/cross-20261013"
ELF = "/mnt/tts/data/cases/52727ec43154/merged.wav"
ELM = "/mnt/tts/data/cases/8197d9ac8a35/merged.wav"

def load16(path, ss=None, to=None):
    t = tempfile.mktemp(suffix=".wav")
    cmd = ["ffmpeg","-y","-v","error"]
    if ss is not None: cmd += ["-ss", str(ss)]
    if to is not None: cmd += ["-to", str(to)]
    cmd += ["-i",path,"-ar","16000","-ac","1",t]
    subprocess.run(cmd, check=True)
    y, sr = sf.read(t, dtype="float32"); os.unlink(t)
    return y, sr

def env(y, sr):
    fl = int(0.01*sr); n = len(y)//fl
    return np.sqrt((y[:n*fl].reshape(n,fl)**2).mean(axis=1))

def align(clip, src, coarse=0.4, fine=0.02):
    """包络相关：返回 clip 起点在 src 中的秒数"""
    c, sr = clip; s, _ = src
    ec, es = env(c, sr), env(s, sr)
    L = min(len(ec), 1500)          # 15s 模板
    tpl = ec[:L] - ec[:L].mean()
    tn = np.linalg.norm(tpl)
    best, bo = -2, None
    for off in range(0, len(es)-L, int(coarse*100)):
        seg = es[off:off+L]; seg = seg - seg.mean()
        d = tn*np.linalg.norm(seg)
        if d < 1e-9: continue
        cc = float((tpl*seg).sum()/d)
        if cc > best: best, bo = cc, off
    # 细化 ±0.5s
    if bo is not None:
        c0, c1 = max(0, bo-int(0.5/coarse*100)), min(len(es)-L, bo+int(0.5/coarse*100))
        for off in range(c0, c1, 2):
            seg = es[off:off+L]; seg = seg - seg.mean()
            d = tn*np.linalg.norm(seg)
            if d < 1e-9: continue
            cc = float((tpl*seg).sum()/d)
            if cc > best: best, bo = cc, off
    return (bo*0.01 if bo is not None else None), best

def runs_gaps(y, sr, thr_db=-40, min_sil=0.04):
    """返回交替的 (类型, 起, 止)：浊音段 / 间隙"""
    fl = int(0.01*sr); n = len(y)//fl
    fr = y[:n*fl].reshape(n, fl)
    db = 20*np.log10(np.sqrt((fr**2).mean(axis=1))+1e-12)
    voiced = db > thr_db
    out, i = [], 0
    while i < n:
        j = i
        while j+1 < n and voiced[j+1] == voiced[i]: j += 1
        out.append(("V" if voiced[i] else "G", round(i*0.01,3), round((j+1)*0.01,3)))
        i = j+1
    # 合并过短间隙
    return [(t,a,b) for (t,a,b) in out if not (t=="G" and b-a < min_sil)]

print("=== A. 号码段对齐与结构对比 ===")
clip = load16(f"{BASE}/bank5/h7_fn_m.mp3")
src_el = load16(ELF)
off, cc = align(clip, src_el)
print(f"我们的片段 h7_fn_m 对齐 EL 5272: 起点={off}s 相关={cc:.3f}")
if off is None: off = 195.7
ours, sr = load16(f"{BASE}/X2f_v3f.wav", ss=195.7, to=227.3)
els, _ = load16(ELF, ss=off, to=off+31.6)
def show(name, y):
    rg = runs_gaps(y, sr)
    print(f"\n{name}:")
    for t,a,b in rg[:60]:
        d=b-a
        if t=="G": print(f"   GAP  {a:6.2f}-{b:6.2f} {d:.3f}s")
    # 只显示浊音段时长序列（前30个）
    vs=[(round(a,2),round(b-a,3)) for t,a,b in rg if t=="V"][:30]
    print("   浊音段(起,时长):", vs)
show("我们 X2f_v3f 号码段 [195.7~227.3]", ours)
show("EL 5272 对应段", els)
