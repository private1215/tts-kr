#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""女声修复 v3（第六轮反馈：「啊」声/断句怪音仍在）：
① 逐 60s 窗音节核速率对齐 EL（与 v2 同法，boost=1.0，语速已被员工认可不再动）；
② 停顿清洗重做：≥0.08s 停顿内容整体替换为「合成干净 room tone」
  （pause_clean_v3.synth_tone：白噪声谱形整形，除 50Hz 嗡、HF 缓降），
  电平=每处停顿自身中段 RMS（保留现有底噪），长度不变，5ms 交叉淡化。
  ⚠️ v2 失败教训：模板取「RMS 最低段」→ 该段本身带谐波凸起（实测全篇
  0 个合格候选），替换后员工仍听到「啊」。v3 不依赖原素材底噪。
③ 响度对齐 EL。
用法: female_fix_v3.py <raw.wav> <el_ref.wav> <out.wav>
"""
import sys, os, subprocess, tempfile
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pause_clean_v3 import clean_gaps, detect_gaps_wav, verify_pauses

WIN = 60.0

def load16(path):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg","-y","-v","error","-i",path,"-ar","16000","-ac","1",t],check=True)
    y, sr = sf.read(t, dtype="float32"); os.unlink(t)
    return y, sr

def syl_rates(y, sr):
    hop = int(0.01*sr)
    n = len(y)//hop
    env = np.sqrt(np.array([(y[i*hop:(i+1)*hop]**2).mean() for i in range(n)]))
    env = np.convolve(env, np.ones(5)/5, mode="same")
    thr = np.percentile(env, 30)*2.2
    w = int(WIN*100)
    out, peaks, voiced = [], 0, 0
    for i in range(n):
        if env[i] > thr: voiced += 1
        if (env[i] > thr and 0 < i < n-1 and env[i] >= env[i-1] and env[i] > env[i+1]
                and env[i]-max(env[max(0,i-8):i].min(),
                               env[i+1:i+9].min() if i+9 < n else 0) > thr*0.35):
            peaks += 1
        if (i+1) % w == 0:
            if voiced > 0: out.append(peaks/(voiced/100.0))
            peaks, voiced = 0, 0
    return out

def detect_gaps(path, thr=-45, mind=0.08):
    return detect_gaps_wav(path, thr=thr, mind=mind)

def lufs(p):
    r = subprocess.run(["ffmpeg","-i",p,"-af","ebur128","-f","null","-"],capture_output=True,text=True)
    ms = re.findall(r"I:\s+(-?[\d.]+)\s+LUFS", r.stderr)
    return float(ms[-1]) if ms else -24.0

import re

def main():
    raw, el, out = sys.argv[1], sys.argv[2], sys.argv[3]
    y, sr = load16(raw)
    ely, _ = load16(el)
    dur_raw, dur_el = len(y)/sr, len(ely)/sr
    rc, ec = syl_rates(y, sr), syl_rates(ely, sr)
    fs = []
    for i, r in enumerate(rc):
        frac = (i+0.5)*WIN/dur_raw
        centers = [(j+0.5)*WIN/dur_el for j in range(len(ec))]
        er = float(np.interp(frac, centers, ec)) if ec else r
        fs.append(min(max(r/er if er>0 else 1.0, 1.0), 1.30))
    fs = [ (fs[i-1]+2*fs[i]+fs[i+1])/4 if 0<i<len(fs)-1 else fs[i] for i in range(len(fs)) ]
    print("raw速率:", [round(x,2) for x in rc])
    print("EL 速率:", [round(x,2) for x in ec])
    print("窗口系数:", [round(x,3) for x in fs], flush=True)

    graw, _ = detect_gaps(raw)
    bounds = [0.0]
    for i in range(1, len(fs)):
        want = i*WIN
        cand = [g for g in graw if abs(g[0]-want) <= 10.0 and g[1] >= 0.08]
        b = min(cand, key=lambda g: abs(g[0]-want))[0] if cand else want
        bounds.append(max(b, bounds[-1]+1.0))
    bounds.append(dur_raw)
    tmpd = tempfile.mkdtemp()
    segs = []
    for i in range(len(fs)):
        a, b = bounds[i], bounds[i+1]
        t1 = os.path.join(tmpd,"in.wav"); t2 = os.path.join(tmpd,"out.wav")
        sf.write(t1, y[int(a*sr):int(b*sr)], sr)
        subprocess.run(["ffmpeg","-y","-v","error","-i",t1,"-af","atempo=%.4f"%(1.0/fs[i]),t2],check=True)
        s2,_ = sf.read(t2, dtype="float32")
        segs.append(s2)
    buf = np.concatenate(segs)
    t3 = os.path.join(tmpd,"st.wav")
    sf.write(t3, buf, sr)
    print("分段拉伸后 %.1fs" % (len(buf)/sr), flush=True)

    # 停顿清洗 v3：合成干净 tone 替换 ≥0.08s 停顿内容
    gs, dur_s = detect_gaps(t3)
    buf, n_clean, floors = clean_gaps(buf, sr, gs, min_dur=0.08, w_ms=5)
    fl_db = 20*np.log10(np.median(floors)+1e-9) if floors else float("nan")
    print("停顿清洗 %d 处（合成tone，底噪中位 %.1f dBFS，长度不变）" % (n_clean, fl_db), flush=True)

    t4 = os.path.join(tmpd,"clean.wav")
    sf.write(t4, buf, sr)
    verify_pauses(t4, "清洗后")
    li, lo = lufs(el), lufs(t4)
    gain = round(li-lo, 1)
    subprocess.run(["ffmpeg","-y","-v","error","-i",t4,"-af","volume=%gdB"%gain,out],check=True)
    yv, srv = load16(out)
    fc = syl_rates(yv, srv)
    gs2, _ = detect_gaps(out)
    ngd = sorted([g for _,g in gs2], reverse=True)
    mid = [g for g in ngd if 0.3<=g<0.8]
    print("修复后速率:", [round(x,2) for x in fc], flush=True)
    verify_pauses(out, "最终")
    print("DONE %s %.1fs 停顿%d处(0.3-0.8s:%d处/%.2fs) LUFS=%.1f(EL %.1f)" % (
        out, len(yv)/srv, len(gs2), len(mid), sum(mid), lufs(out), li), flush=True)

if __name__ == "__main__":
    main()
