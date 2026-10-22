# -*- coding: utf-8 -*-
"""slowdyn 纯函数自检：只看它自己的输入输出，不模拟链路（避免陷阱 #17）。"""
import os, sys, numpy as np
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")
from app import slowdyn
from app.prosody_report import sent_windows
from tools.drift import load, SR

def levels(x, sr, segs):
    """与 slowdyn.sent_levels 同源，但返回原始序列用于前后对比。"""
    return slowdyn.sent_levels(x, sr, segs)

def ac1(v):
    a = v - v.mean()
    return float(np.sum(a[:-1]*a[1:])/max(np.sum(a*a),1e-12))

def ma_share(v, k=10):
    c = np.concatenate([[0.0], np.cumsum(v)])
    m = (c[k:]-c[:-k])/float(k)
    return float(m.std()/max(v.std(),1e-9)), float(m.std()), float(v.std())

for name, path, male in [
    ("r25a-1a6c20 男", "/mnt/tts/output/r25a-1a6c20", True),
]:
    import glob
    import subprocess, tempfile, soundfile as sf
    fs = sorted(glob.glob(os.path.join(path, "*.mp3")))
    if not fs:
        print("缺产物", path); continue
    lst = "/tmp/_c.txt"
    open(lst,"w").write("".join("file '%s'\n"%f for f in fs))
    out = "/tmp/_c.wav"
    subprocess.run(["ffmpeg","-nostdin","-v","error","-y","-f","concat","-safe","0",
                    "-i",lst,"-ac","1","-ar",str(SR),out],check=True)
    x = np.asarray(sf.read(out,dtype="float32")[0],dtype=np.float32)
    segs,_,_ = sent_windows(x, SR)
    print(f"\n{name}  {len(x)/SR:.1f}s  {len(segs)} 句")
    d0, spans = slowdyn.sent_levels(x, SR, segs)
    s0 = slowdyn.stats_of(d0)
    print(f"  处理前: 句级sd {s0['sd']:.3f}  慢变sd {s0['sd_s']:.3f}  "
          f"占比 {s0['share']:.3f}  ac1 {ac1(d0):+.3f}")
    sh0, ms0, sd0 = ma_share(d0)
    print(f"  处理前 30s窗等效(MA10 sd) {ms0:.3f} dB")
    print(f"  目标:  句级sd {slowdyn._sd_target(male):.3f}  "
          f"占比 {slowdyn._slow_target(male):.3f}")
    ks, kf, st = slowdyn.solve(d0, male)
    print(f"  反解: k_slow={ks:.3f} k_fast={kf:.3f}")
    y, ks2, kf2, diag = slowdyn.apply(x, SR, segs=segs, is_male=male)
    segs2,_,_ = sent_windows(y, SR)
    d1, _ = slowdyn.sent_levels(y, SR, segs2)
    s1 = slowdyn.stats_of(d1)
    print(f"  处理后: 句级sd {s1['sd']:.3f}  慢变sd {s1['sd_s']:.3f}  "
          f"占比 {s1['share']:.3f}  ac1 {ac1(d1):+.3f}")
    sh1, ms1, sd1 = ma_share(d1)
    print(f"  处理后 30s窗等效(MA10 sd) {ms1:.3f} dB   "
          f"→ 倍数 {ms1/max(ms0,1e-9):.2f}")
    print(f"  增益范围 {diag['gain_min']:+.2f} ~ {diag['gain_max']:+.2f} dB")
    r0 = np.sqrt(np.mean(x.astype(np.float64)**2))
    r1 = np.sqrt(np.mean(y.astype(np.float64)**2))
    print(f"  整段 RMS: {20*np.log10(r0):.3f} -> {20*np.log10(r1):.3f} dB "
          f"(差 {20*np.log10(r1/r0):+.4f})")
    p0 = np.max(np.abs(x)); p1 = np.max(np.abs(y))
    print(f"  真峰值: {20*np.log10(p0):.2f} -> {20*np.log10(p1):.2f} dBFS "
          f"(差 {20*np.log10(p1/p0):+.2f})")
