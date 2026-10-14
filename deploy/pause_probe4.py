#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pause_probe4.py — 铁证验证：
1. 女声 v2 的 30 处脏停顿是否同一素材（互相相关）
2. 男声 v4 极端脏停顿（max_above>40dB）位置明细
3. 模板候选段扫描：全篇找真正干净的 room tone（谱平坦度高、无谐波、无瞬态）
"""
import numpy as np, soundfile as sf, subprocess, re, json

BASE = "/mnt/tts/diag/cross-20261013"

def load(path):
    data, fs = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, fs

def find_pauses(path, thr_db=-45, min_dur=0.15):
    cmd = ["ffmpeg", "-nostats", "-i", path,
           "-af", f"silencedetect=noise={thr_db}dB:d={min_dur}", "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    return [(s, ends[i] if i < len(ends) else None) for i, s in enumerate(starts)]

def spec_flat(seg, fs):
    w = np.hanning(len(seg))
    sp = np.abs(np.fft.rfft(seg * w)) ** 2
    gm = np.exp(np.mean(np.log(sp + 1e-12)))
    return float(gm / (np.mean(sp) + 1e-12))

print("=== 1. 女声 v2 脏停顿同一性验证 ===")
data, fs = load(f"{BASE}/X2f_v2.wav")
pauses = find_pauses(f"{BASE}/X2f_v2.wav")
segs = []
for (s, e) in pauses:
    if e is None:
        continue
    i0, i1 = int(s * fs), int(e * fs)
    seg = data[i0:i1]
    fl = int(0.01 * fs)
    n = len(seg) // fl
    if n < 5:
        continue
    fr = seg[:n * fl].reshape(n, fl)
    rms = np.sqrt((fr ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    med = np.median(db)
    if db.max() - med > 8:
        # 取能量凸起处 ±60ms 做互相关
        pk = int(np.argmax(db)) * fl
        c0 = max(0, pk - int(0.06 * fs))
        c1 = min(len(seg), pk + int(0.06 * fs))
        segs.append((s, seg[c0:c1].copy()))
print(f"提取到 {len(segs)} 个凸起片段")
if len(segs) >= 3:
    # 统一长度后互相关
    L = min(len(x[1]) for x in segs)
    mat = np.stack([x[1][:L] for x in segs])
    mat = mat - mat.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    matn = mat / norm
    cc = matn @ matn.T
    iu = np.triu_indices(len(segs), 1)
    print(f"两两相关: 均值={cc[iu].mean():.3f} 中位={np.median(cc[iu]):.3f} 最大={cc[iu].max():.3f} 最小={cc[iu].min():.3f}")
    print(f"相关>0.8 的pair占比: {(cc[iu]>0.8).mean()*100:.0f}%")
    # 平均波形能量谱
    avg = mat.mean(axis=0)
    sp = np.abs(np.fft.rfft(avg * np.hanning(len(avg)))) ** 2
    freqs = np.fft.rfftfreq(len(avg), 1 / fs)
    top = np.argsort(sp)[-8:][::-1]
    print("平均凸起的主频峰(Hz):", [(round(freqs[i]), round(10 * np.log10(sp[i] + 1e-12), 1)) for i in top])

print()
print("=== 2. 男声 v4 极端脏停顿 ===")
d2, fs2 = load(f"{BASE}/X1m_v3.wav")  # v4 内容
p2 = find_pauses(f"{BASE}/X1m_v3.wav")
ext = []
for (s, e) in p2:
    if e is None:
        continue
    i0, i1 = int(s * fs2), int(e * fs2)
    seg = d2[i0:i1]
    fl = int(0.01 * fs2)
    n = len(seg) // fl
    if n < 5:
        continue
    fr = seg[:n * fl].reshape(n, fl)
    rms = np.sqrt((fr ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    med = np.median(db)
    above = db.max() - med
    if above > 40:
        pk = int(np.argmax(db)) * fl
        c0, c1 = max(0, pk - int(0.06 * fs2)), min(len(seg), pk + int(0.06 * fs2))
        bump = seg[c0:c1]
        ext.append((s, e - s, above, spec_flat(bump, fs2)))
print(f"极端脏停顿 {len(ext)} 处（max_above>40dB）:")
for s, du, ab, fl_ in ext:
    print(f"  t={s:7.2f} dur={du:.3f} above={ab:5.1f} spec_flat={fl_:.4f}")

print()
print("=== 3. 模板候选段扫描：全篇真正干净的 room tone（女声 raw）===")
dr, fsr = load(f"{BASE}/X2_f.wav")
win = int(0.5 * fsr)
cands = []
for off in range(0, len(dr) - win, win // 2):
    seg = dr[off:off + win]
    rmsdb = 20 * np.log10(np.sqrt((seg ** 2).mean()) + 1e-12)
    fl_ = spec_flat(seg, fsr)
    # 帧级稳定性
    fl10 = int(0.01 * fsr)
    n = len(seg) // fl10
    fr = seg[:n * fl10].reshape(n, fl10)
    rms10 = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-12)
    stab = rms10.max() - np.median(rms10)
    if rmsdb < -50 and fl_ > 0.3 and stab < 6:
        cands.append((off / fsr, round(rmsdb, 1), round(fl_, 3), round(stab, 1)))
print(f"合格候选段（<-50dB 且谱平坦>0.3 且稳定）: {len(cands)}")
for c in cands[:15]:
    print(f"  t={c[0]:7.1f} rms={c[1]}dB flat={c[2]} stab={c[3]}")
