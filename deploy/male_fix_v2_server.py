# -*- coding: utf-8 -*-
"""男声修复 v2（对应员工第四轮两条意见）：
①「前面舒服后面飞快」→ 逐 60s 窗测量音节核速率，对齐 EL 同文本位置速率，
  分段 atempo（切点选静音处，变速不可闻），替代整体均匀拉伸；
②「断句掺杂别的声音」→ 停顿插入素材改为全篇 RMS 最低的干净 room tone 统一模板，
  按目标停顿 RMS 匹配幅度 + 20ms 交叉淡化。
用法: male_fix_v2.py <raw.wav> <el_ref.wav> <out_prefix> <pause_targets_csv>
产出: <out_prefix>_v3.wav，并打印修复前后逐窗速率与停顿指标。
"""
import sys, os, re, json, subprocess, tempfile
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

def interp_rate(rates, durs, frac):
    centers = [ (j+0.5)*WIN/durs for j in range(len(rates)) ]
    return float(np.interp(frac, centers, rates))

def detect_gaps(path):
    r = subprocess.run(["ffmpeg","-i",path,"-af","silencedetect=noise=-45dB:d=0.08",
                        "-f","null","-"],capture_output=True,text=True)
    txt = r.stderr
    s=[float(x) for x in re.findall(r"silence_start: ([\d.]+)",txt)]
    e=[float(x) for x in re.findall(r"silence_end: ([\d.]+)",txt)]
    d=float(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
        "-of","csv=p=0",path],capture_output=True,text=True).stdout)
    return [(a,b-a) for a,b in zip(s,e) if a>1.0 and b<d-1.0], d

def main():
    raw, el, prefix, tcsv = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    boost = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    targets = sorted([float(x) for x in tcsv.split(",")], reverse=True)

    y, sr = load16(raw)
    ely, _ = load16(el)
    dur_raw, dur_el = len(y)/sr, len(ely)/sr
    rc = syl_rates(y, sr); ec = syl_rates(ely, sr)
    # 每窗修正系数
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
    print("每窗拉伸系数:", [round(x,3) for x in fs], flush=True)

    # 在静音处切窗边界
    gaps, _ = detect_gaps(raw)
    bounds = [0.0]
    for i in range(1, len(fs)):
        want = i*WIN
        cand = [g for g in gaps if abs(g[0]-want) <= 10.0 and g[1] >= 0.08]
        if cand:
            b = min(cand, key=lambda g: abs(g[0]-want))[0]
        else:
            b = want
        bounds.append(max(b, bounds[-1]+1.0))
    bounds.append(dur_raw)

    # 分段 atempo
    segs = []
    tmpd = tempfile.mkdtemp()
    for i in range(len(fs)):
        a, b = bounds[i], bounds[i+1]
        seg = y[int(a*sr):int(b*sr)]
        at = 1.0/fs[i]
        t1 = os.path.join(tmpd, "in.wav"); t2 = os.path.join(tmpd, "out.wav")
        sf.write(t1, seg, sr)
        subprocess.run(["ffmpeg","-y","-v","error","-i",t1,"-af","atempo=%.4f"%at,t2],check=True)
        seg2, _ = sf.read(t2, dtype="float32")
        segs.append(seg2)
    buf = np.concatenate(segs)
    t3 = os.path.join(tmpd, "st.wav")
    sf.write(t3, buf, sr)
    print("分段拉伸后 %.1fs" % (len(buf)/sr), flush=True)

    # 停顿整形：干净 room tone 模板
    gaps2, dur2 = detect_gaps(t3)
    gaps2.sort(key=lambda x:-x[1])
    room, room_rms = None, None
    for gs_, gd in gaps2:
        if gd < 0.25: continue
        mid = y_ = None
        a = int((gs_+gd*0.3)*sr); b = int((gs_+gd*0.8)*sr)
        if b-a < int(0.2*sr): continue
        seg = buf[a:b]
        r_ = float(np.sqrt((seg.astype(np.float64)**2).mean()))
        if room_rms is None or r_ < room_rms:
            room_rms, room = r_, seg
    plan = []
    for i, t in enumerate(targets):
        if i >= len(gaps2): break
        gs_, gd = gaps2[i]
        ext = t - gd
        if ext <= 0.02 or room is None: continue
        # 目标停顿的 RMS（匹配幅度）
        a = int((gs_+gd*0.25)*sr); b_ = min(int((gs_+gd*0.9)*sr), len(buf))
        trms = float(np.sqrt((buf[a:b_].astype(np.float64)**2).mean())) if b_-a > 64 else room_rms
        gain = trms/(room_rms+1e-9)
        need = int(ext*sr)
        reps = int(np.ceil(need/len(room)))
        ins = np.tile(room, reps)[:need]*gain
        plan.append((gs_+gd*0.45, ins.astype("float32")))
    plan.sort(key=lambda x:-x[0])
    for pos, ins in plan:
        p = int(pos*sr); w = int(0.02*sr)
        if len(ins) > 2*w:
            buf = np.concatenate([buf[:p+w], ins, buf[p+w:]])
            ramp = np.linspace(0,1,w)
            buf[p:p+w] = buf[p:p+w]*(1-ramp)+ins[:w]*ramp
            n = len(ins)
            buf[p+n-w:p+n] = buf[p+n-w:p+n]*(1-ramp[::-1])+ins[-w:]*ramp[::-1]
        else:
            buf = np.concatenate([buf[:p], ins, buf[p:]])
    t4 = prefix + "_v3_raw.wav"
    sf.write(t4, buf, sr)
    print("停顿整形后 %.1fs（插入 %d 处）" % (len(buf)/sr, len(plan)), flush=True)

    # 压亮 EQ + 响度对齐
    def lufs(p):
        r = subprocess.run(["ffmpeg","-i",p,"-af","ebur128","-f","null","-"],capture_output=True,text=True)
        ms = re.findall(r"I:\s+(-?[\d.]+)\s+LUFS", r.stderr)
        return float(ms[-1]) if ms else -24.0
    li, lo = lufs(el), lufs(t4)
    gain = round(li-lo, 1)
    out = prefix + "_v3.wav"
    subprocess.run(["ffmpeg","-y","-v","error","-i",t4,"-af",
                    "highshelf=f=4000:g=-4,lowpass=f=14000,volume=%gdB"%gain,out],check=True)
    # 复核
    yv, srv = load16(out)
    durv = len(yv)/srv
    fc = syl_rates(yv, srv)
    print("修复后速率:", [round(x,2) for x in fc], flush=True)
    gaps3, dur3 = detect_gaps(out)
    ngd = sorted([g for _,g in gaps3], reverse=True)
    mid = [g for g in ngd if 0.3<=g<0.8]
    print("DONE %s %.1fs 长停顿%d处/%.2fs top8=%s LUFS=%.1f(EL %.1f)" % (
        out, durv, len(mid), sum(mid), [round(x,2) for x in ngd[:8]],
        lufs(out), li), flush=True)

if __name__ == "__main__":
    main()
