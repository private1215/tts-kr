#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""男声修复 v3 → 产出 v5（第六轮反馈：「停顿处有怪声音」+ 语速回到 v3）：
① 逐 60s 窗音节核速率对齐 EL，boost=1.0（= v3 语速，员工第五轮裁定
  「v3 比 v4 更接近目标」，v4 的 +3% 矫枉过正，回退）；
② 停顿清洗重做（核心修复）：v4 及以前根本没清洗停顿（v4 日志「插入 0 处」，
  atempo/EQ 不动停顿内容）——实测 VoxCPM2 原始输出 64 处停顿里 60 处脏
  （能量凸起 8-70dB，含语音电平残留）。v5 用 pause_clean_v3 的合成干净
  tone 整体替换 ≥0.08s 停顿内容（长度不变、底噪电平保留、5ms 交叉淡化）；
③ 停顿插入目标传 "none" 跳过（atempo 拉伸后长停顿已达 15 处，对齐 EL）；
④ 压亮 EQ + 响度对齐（同 v2/v4）。
用法: male_fix_v3.py <raw.wav> <el_ref.wav> <out_prefix> none [boost]
产出: <out_prefix>_v5.wav / _v5_raw.wav
"""
import sys, os, re, subprocess, tempfile
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

def interp_rate(rates, durs, frac):
    centers = [ (j+0.5)*WIN/durs for j in range(len(rates)) ]
    return float(np.interp(frac, centers, rates))

def detect_gaps(path):
    return detect_gaps_wav(path, thr=-45, mind=0.08)

def lufs(p):
    r = subprocess.run(["ffmpeg","-i",p,"-af","ebur128","-f","null","-"],capture_output=True,text=True)
    ms = re.findall(r"I:\s+(-?[\d.]+)\s+LUFS", r.stderr)
    return float(ms[-1]) if ms else -24.0

def main():
    raw, el, prefix, tcsv = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    boost = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    do_insert = tcsv.strip().lower() not in ("none", "")
    targets = sorted([float(x) for x in tcsv.split(",")], reverse=True) if do_insert else []

    y, sr = load16(raw)
    ely, _ = load16(el)
    dur_raw, dur_el = len(y)/sr, len(ely)/sr
    rc = syl_rates(y, sr); ec = syl_rates(ely, sr)
    fs = []
    for i, r in enumerate(rc):
        frac = (i+0.5)*WIN/dur_raw
        er = interp_rate(ec, dur_el, frac)
        f = r/er if er > 0 else 1.0
        fs.append(min(max(f, 1.0), 1.30))
    fs = [ (fs[i-1]+2*fs[i]+fs[i+1])/4 if 0<i<len(fs)-1 else fs[i] for i in range(len(fs)) ]
    fs = [min(max(f*boost, 1.0), 1.35) for f in fs]
    print("raw速率:", [round(x,2) for x in rc])
    print("EL 速率:", [round(x,2) for x in ec])
    print("每窗拉伸系数(boost=%g):"%boost, [round(x,3) for x in fs], flush=True)

    gaps, _ = detect_gaps(raw)
    bounds = [0.0]
    for i in range(1, len(fs)):
        want = i*WIN
        cand = [g for g in gaps if abs(g[0]-want) <= 10.0 and g[1] >= 0.08]
        b = min(cand, key=lambda g: abs(g[0]-want))[0] if cand else want
        bounds.append(max(b, bounds[-1]+1.0))
    bounds.append(dur_raw)

    segs = []
    tmpd = tempfile.mkdtemp()
    for i in range(len(fs)):
        a, b = bounds[i], bounds[i+1]
        seg = y[int(a*sr):int(b*sr)]
        t1 = os.path.join(tmpd,"in.wav"); t2 = os.path.join(tmpd,"out.wav")
        sf.write(t1, seg, sr)
        subprocess.run(["ffmpeg","-y","-v","error","-i",t1,"-af","atempo=%.4f"%(1.0/fs[i]),t2],check=True)
        s2, _ = sf.read(t2, dtype="float32")
        segs.append(s2)
    buf = np.concatenate(segs)
    t3 = os.path.join(tmpd,"st.wav")
    sf.write(t3, buf, sr)
    print("分段拉伸后 %.1fs" % (len(buf)/sr), flush=True)
    verify_pauses(t3, "拉伸后(清洗前)")

    # 停顿清洗 v5：合成干净 tone 替换 ≥0.08s 停顿（核心修复）
    gs2, dur2 = detect_gaps(t3)
    buf, n_clean, floors = clean_gaps(buf, sr, gs2, min_dur=0.08, w_ms=5)
    fl_db = 20*np.log10(np.median(floors)+1e-9) if floors else float("nan")
    print("停顿清洗 %d 处（合成tone，底噪中位 %.1f dBFS，长度不变）" % (n_clean, fl_db), flush=True)

    # 停顿插入 targets=none 时跳过（atempo 拉伸后长停顿数量已对齐 EL；
    # 如未来需要插入，用 pause_clean_v3.synth_tone 生成内容再复清洗）
    t4 = prefix + "_v5_raw.wav"
    sf.write(t4, buf, sr)
    verify_pauses(t4, "清洗后")

    def lufs_(p):
        return lufs(p)
    li, lo = lufs_(el), lufs_(t4)
    gain = round(li-lo, 1)
    out = prefix + "_v5.wav"
    subprocess.run(["ffmpeg","-y","-v","error","-i",t4,"-af",
                    "highshelf=f=4000:g=-4,lowpass=f=14000,volume=%gdB"%gain,out],check=True)
    yv, srv = load16(out)
    durv = len(yv)/srv
    fc = syl_rates(yv, srv)
    print("修复后速率:", [round(x,2) for x in fc], flush=True)
    gaps3, dur3 = detect_gaps(out)
    ngd = sorted([g for _,g in gaps3], reverse=True)
    mid = [g for g in ngd if 0.3<=g<0.8]
    verify_pauses(out, "最终")
    print("DONE %s %.1fs 长停顿%d处/%.2fs top8=%s LUFS=%.1f(EL %.1f)" % (
        out, durv, len(mid), sum(mid), [round(x,2) for x in ngd[:8]],
        lufs(out), li), flush=True)

if __name__ == "__main__":
    main()
