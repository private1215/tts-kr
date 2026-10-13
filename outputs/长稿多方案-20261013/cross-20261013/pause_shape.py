# -*- coding: utf-8 -*-
"""停顿整形：把最长若干停顿延长到 EL 分布（呼吸长停顿），
插入素材 = 停顿自带底噪复制 + 20ms 等功率交叉淡化（避免静音落差）。
用法: pause_shape.py <in.wav> <out.wav> <targets_csv>
  targets_csv = 目标时长列表(秒,降序)，如 "0.55,0.52,0.50,0.48,0.46,0.44,0.40,0.38,0.36,0.34,0.32"
第 i 个目标对应第 i 长的停顿；只延长不缩短。
"""
import sys, re, subprocess
import numpy as np
import soundfile as sf

def detect_gaps(path, total):
    r = subprocess.run(["ffmpeg","-i",path,"-af","silencedetect=noise=-45dB:d=0.08",
                        "-f","null","-"],capture_output=True,text=True)
    txt=r.stderr
    s=[float(x) for x in re.findall(r"silence_start: ([\d.]+)",txt)]
    e=[float(x) for x in re.findall(r"silence_end: ([\d.]+)",txt)]
    g=[(a,b-a) for a,b in zip(s,e) if a>1.0 and b<total-1.0]
    return g

def xfade_insert(buf, pos, insert, sr):
    """在 pos(样本) 后插入 insert，两端 20ms 等功率交叉淡化。"""
    w = int(0.02*sr)
    n = len(insert)
    if n <= 2*w:  # 过短插入直接硬接（底噪连续，落差可忽略）
        return np.concatenate([buf[:pos], insert, buf[pos:]])
    # 右移 pos 后面的内容
    out = np.concatenate([buf[:pos+w],
                          np.zeros(n, dtype=buf.dtype),
                          buf[pos+w:]])
    ins = insert.copy()
    # 左淡入与原信号混合
    ramp = np.linspace(0,1,w)
    out[pos:pos+w] = out[pos:pos+w]*(1-ramp) + ins[:w]*ramp
    # 中段
    out[pos+w:pos+w+n-2*w] = ins[w:-w] if n>2*w else ins[w:n//2]
    # 右淡出与插入尾部后的原内容混合
    if n > 2*w:
        out[pos+n-w:pos+n] = out[pos+n-w:pos+n]*(1-ramp[::-1]) + ins[-w:]*ramp[::-1]
    return out

def main():
    inp, outp, tcsv = sys.argv[1], sys.argv[2], sys.argv[3]
    targets = sorted([float(x) for x in tcsv.split(",")], reverse=True)
    data, sr = sf.read(inp, dtype="float32")
    if data.ndim > 1: data = data.mean(axis=1)
    total = len(data)/sr
    gaps = detect_gaps(inp, total)
    gaps.sort(key=lambda x:-x[1])
    room = None  # 底噪模板：取全篇最长停顿中段
    if gaps:
        gs, gd = gaps[0]
        a, b = int((gs+gd*0.3)*sr), int((gs+gd*0.8)*sr)
        if b-a > int(0.2*sr): room = data[a:b]
    plan = []
    for i, t in enumerate(targets):
        if i >= len(gaps): break
        gs, gd = gaps[i]
        ext = t - gd
        if ext <= 0.02: continue
        src_a = int((gs + gd*0.25)*sr)
        src_b = int((gs + gd*0.9)*sr)
        if src_b - src_a < int(0.1*sr) or src_b > len(data):
            if room is None: continue
            seg = room
        else:
            seg = data[src_a:src_b]
        if len(seg) < int(0.12*sr) and room is not None:
            seg = room
        need = int(ext*sr)
        if len(seg) >= need:
            ins = seg[:need].copy()
        else:
            reps = int(np.ceil(need/len(seg)))
            ins = np.tile(seg, reps)[:need].copy()
        plan.append((gs + gd*0.45, ins))  # 插入点=停顿中段
    plan.sort(key=lambda x:-x[0])  # 从后往前插
    buf = data.copy()
    for pos, ins in plan:
        buf = xfade_insert(buf, int(pos*sr), ins.astype("float32"), sr)
    sf.write(outp, buf, sr)
    # 复核
    r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                        "-of","csv=p=0",outp],capture_output=True,text=True)
    ng = detect_gaps(outp, float(r.stdout))
    ngd = sorted([d for _,d in ng], reverse=True)
    mid = [d for d in ngd if 0.3<=d<0.8]
    print("OUT %s dur=%.1fs gaps=%d 0.3-0.8s=%d处/%.2fs top10=%s" % (
        outp, float(r.stdout), len(ng), len(mid), sum(mid),
        [round(x,2) for x in ngd[:10]]), flush=True)

if __name__ == "__main__":
    main()
