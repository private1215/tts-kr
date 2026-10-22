# -*- coding: utf-8 -*-
"""判据复核：句间能量负相关是"真的抖动"还是"判句长度交替"的伪影。"""
import os, sys, numpy as np
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")
from app.prosody_report import sent_windows
from tools.drift import load, SR, CASES_DIR, WORK

def seq(x, sr):
    segs, _, _ = sent_windows(x, sr)
    db, dur, vrat = [], [], []
    for a, b in segs:
        s = x[a:b]
        if len(s) < int(0.15*sr): continue
        db.append(20*np.log10(np.sqrt(np.mean(s**2))+1e-12))
        dur.append((b-a)/float(sr))
        # 句内有声占比（相对阈值），把"句里有沉默拉低均值"这个因素剥掉
        W,H = int(0.05*sr), int(0.025*sr)
        n = (len(s)-W)//H
        if n > 5:
            e = np.sqrt(np.array([np.mean(s[i*H:i*H+W]**2) for i in range(n)])+1e-12)
            dbe = 20*np.log10(e+1e-12)
            thr = np.percentile(dbe,25)+6.0
            v = dbe > thr
            vrat.append(v.mean())
            if v.sum() > 2:
                db[-1] = 20*np.log10(np.sqrt(np.mean(e[v]**2))+1e-12)  # 只算有声帧
            else:
                db[-1] = np.nan
        else:
            vrat.append(np.nan)
    return np.array(db), np.array(dur), np.array(vrat)

def ac1(v):
    v = v[~np.isnan(v)]
    a = v - v.mean()
    return float(np.sum(a[:-1]*a[1:])/max(np.sum(a*a),1e-12))

def corr(a,b):
    m = ~(np.isnan(a)|np.isnan(b))
    if m.sum() < 10: return float('nan')
    return float(np.corrcoef(a[m],b[m])[0,1])

for short, cid in [("1a6c20","1a6c20a60481"),("ba24c0","ba24c0925311"),
                   ("8197d9","8197d9ac8a35")]:
    pw = os.path.join(WORK, f"r23_{short}.wav")
    px = load(pw, SR); cx = load(os.path.join(CASES_DIR,cid,"merged.wav"), SR)
    for tag, x in (("我们",px),("案例",cx)):
        db, dur, vr = seq(x, SR)
        print(f"{short} {tag}: 句数{len(db)}  有声段电平 ac1 {ac1(db):+.2f}"
              f"   句长 ac1 {ac1(dur):+.2f}   corr(电平,句长) {corr(db,dur):+.2f}"
              f"   corr(电平,有声占比) {corr(db,vr):+.2f}"
              f"   句长均值{dur.mean():.2f}s")
