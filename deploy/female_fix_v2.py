# -*- coding: utf-8 -*-
"""女声修复 v2（针对第五轮反馈）：
①「结尾语速太快」→ 逐 60s 窗音节核速率对齐 EL（与男声 v2/v4 同法），静音处切窗；
②「断句夹『啊』声」→ 停顿清洗：把 ≥0.15s 的停顿内容整体替换为全篇最干净的
   room tone 模板（长度不变、边缘 20ms 交叉淡化）——从根上消灭元音碎片；
③不动停顿长度（上轮教训：延长停顿反而被否），不加 EQ（女声频谱已对齐）；
④响度对齐 EL（LUFS 差值补偿）。
用法: female_fix_v2.py <raw.wav> <el_ref.wav> <out.wav> [boost]
"""
import sys, os, re, subprocess, tempfile
import numpy as np
import soundfile as sf

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
    r = subprocess.run(["ffmpeg","-i",path,"-af","silencedetect=noise=%sdB:d=%f"%(thr,mind),
                        "-f","null","-"],capture_output=True,text=True)
    txt=r.stderr
    s=[float(x) for x in re.findall(r"silence_start: ([\d.]+)",txt)]
    e=[float(x) for x in re.findall(r"silence_end: ([\d.]+)",txt)]
    d=float(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
        "-of","csv=p=0",path],capture_output=True,text=True).stdout)
    return [(a,b-a) for a,b in zip(s,e) if a>1.0 and b<d-1.0], d

def lufs(p):
    r = subprocess.run(["ffmpeg","-i",p,"-af","ebur128","-f","null","-"],capture_output=True,text=True)
    ms = re.findall(r"I:\s+(-?[\d.]+)\s+LUFS", r.stderr)
    return float(ms[-1]) if ms else -24.0

def main():
    raw, el, out = sys.argv[1], sys.argv[2], sys.argv[3]
    boost = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

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
    fs = [min(max(f*boost, 1.0), 1.35) for f in fs]
    print("raw速率:", [round(x,2) for x in rc])
    print("EL 速率:", [round(x,2) for x in ec])
    print("窗口系数:", [round(x,3) for x in fs], flush=True)

    # 静音处切窗分段 atempo
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

    # 停顿清洗：替换 ≥0.15s 停顿内容为干净模板（长度不变）
    gs, dur_s = detect_gaps(t3)
    # 模板 = RMS 最低的停顿中段（≥0.2s 的停顿里选）
    room, room_rms = None, None
    for a, dd in sorted(gs, key=lambda x:-x[1]):
        if dd < 0.20: continue
        m = buf[int((a+dd*0.3)*sr):int((a+dd*0.8)*sr)]
        if len(m) < int(0.15*sr): continue
        r_ = float(np.sqrt((m.astype(np.float64)**2).mean()))
        if room_rms is None or r_ < room_rms:
            room, room_rms = m, r_
    n_clean = 0
    if room is not None:
        w = int(0.02*sr)
        ramp = np.linspace(0,1,w)
        for a, dd in gs:
            if dd < 0.15: continue
            i0, i1 = int(a*sr), int((a+dd)*sr)
            if i1-i0 < 2*w or i1 > len(buf): continue
            reps = int(np.ceil((i1-i0)/len(room)))
            ins = (np.tile(room, reps)[:i1-i0]).astype("float32")
            # 边缘交叉淡化（左：原→模板，右：模板→原）
            buf[i0:i0+w] = buf[i0:i0+w]*(1-ramp) + ins[:w]*ramp
            buf[i0+w:i1-w] = ins[w:-w]
            buf[i1-w:i1] = ins[-w:]*(1-ramp[::-1]) + buf[i1-w:i1]*ramp[::-1]
            n_clean += 1
    print("停顿清洗 %d 处（模板RMS %.1f dBFS）" % (n_clean, 20*np.log10(room_rms+1e-9)), flush=True)

    t4 = os.path.join(tmpd,"clean.wav")
    sf.write(t4, buf, sr)
    # 响度对齐
    li, lo = lufs(el), lufs(t4)
    gain = round(li-lo, 1)
    subprocess.run(["ffmpeg","-y","-v","error","-i",t4,"-af","volume=%gdB"%gain,out],check=True)
    # 复核
    yv, srv = load16(out)
    fc = syl_rates(yv, srv)
    gs2, _ = detect_gaps(out)
    ngd = sorted([g for _,g in gs2], reverse=True)
    mid = [g for g in ngd if 0.3<=g<0.8]
    print("修复后速率:", [round(x,2) for x in fc], flush=True)
    print("DONE %s %.1fs 停顿%d处(0.3-0.8s:%d处/%.2fs) LUFS=%.1f(EL %.1f)" % (
        out, len(yv)/srv, len(gs2), len(mid), sum(mid), lufs(out), li), flush=True)

if __name__ == "__main__":
    main()
