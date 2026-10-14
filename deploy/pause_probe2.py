#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pause_probe2.py — 停顿处怪音特征定位。
对比处理前后每处停顿的声学特征，区分：
  - 残留音节（谐波结构 + 能量凸起）→ TTS 真念了东西 / 清洗没覆盖
  - 边界咔哒（首尾帧瞬态）     → 替换/交叉淡化引入
  - 底噪起伏（宽带 + 能量凸起）  → room tone 模板不匹配
  - clean（平稳低能量）          → 正常
"""
import numpy as np, soundfile as sf, subprocess, re, json, sys
from collections import Counter

def find_pauses(path, thr_db=-45, min_dur=0.15):
    cmd = ["ffmpeg", "-nostats", "-i", path,
           "-af", f"silencedetect=noise={thr_db}dB:d={min_dur}",
           "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r.stderr)]
    ends   = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    out = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else None
        out.append((s, e))
    return out

def analyze(path, label):
    data, fs = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    pauses = find_pauses(path)
    rows = []
    for (s, e) in pauses:
        if e is None:
            e = min(s + 1.0, len(data) / fs)
        i0, i1 = int(s * fs), int(e * fs)
        seg = data[i0:i1]
        fl = int(0.01 * fs)
        n = len(seg) // fl
        if n < 5:
            continue
        fr = seg[:n * fl].reshape(n, fl)
        rms = np.sqrt((fr ** 2).mean(axis=1) + 1e-12)
        db = 20 * np.log10(rms + 1e-12)
        med = float(np.median(db))
        max_above = float(db.max()) - med
        edge = float(max(db[:3].max(), db[-3:].max())) - med
        # 谱平坦度取中段 60%
        m0, m1 = int(n * 0.2), int(n * 0.8)
        if m1 - m0 < 2:
            continue
        mid = seg[m0 * fl:m1 * fl]
        w = np.hanning(len(mid))
        sp = np.abs(np.fft.rfft(mid * w)) ** 2
        gm = float(np.exp(np.mean(np.log(sp + 1e-12))))
        am = float(np.mean(sp))
        flat = gm / (am + 1e-12)
        rows.append(dict(t=round(s, 2), dur=round(e - s, 3),
                         med_db=round(med, 1),
                         max_above=round(max_above, 1),
                         edge_above=round(edge, 1),
                         spec_flat=round(flat, 4)))
    for r in rows:
        tags = []
        if r["edge_above"] > 8:
            tags.append("click?")
        if r["max_above"] > 8 and r["spec_flat"] < 0.02:
            tags.append("syllable?")
        elif r["max_above"] > 8 and r["spec_flat"] >= 0.02:
            tags.append("modulation?")
        r["tags"] = ",".join(tags) if tags else "clean"
    return dict(label=label, path=path, n_pauses=len(rows), rows=rows)

if __name__ == "__main__":
    import os
    base = "/mnt/tts/diag/cross-20261013"
    files = [
        ("F_raw",  f"{base}/X2_f.wav"),     # OmniVoice 裸推理 女5272
        ("F_v2",   f"{base}/X2f_v2.wav"),   # 停顿清洗版（员工听的）
        ("F_v2b",  f"{base}/X2f_v2b.wav"),  # 长停顿补齐版
        ("F_v2st", f"{base}/X2f_st.wav"),   # 中间态（仅逐窗atempo?）
        ("M_raw",  f"{base}/X1_m.wav"),     # VoxCPM2 裸推理 男8197
        ("M_v4",   f"{base}/X1m_v3.wav"),   # 文件名 v3 内容为 v4（员工听的）
        ("M_v2",   f"{base}/X1m_fix.wav"),  # v2 修复版
        ("M_st",   f"{base}/X1m_st.wav"),   # 中间态
    ]
    res = []
    for label, p in files:
        if not os.path.exists(p):
            print(f"[skip] {label} {p} 不存在")
            continue
        r = analyze(p, label)
        res.append(r)
        c = Counter(x["tags"] for x in r["rows"])
        dirty = sum(v for k, v in c.items() if k != "clean")
        print(f"{label:8s} pauses={r['n_pauses']:3d} dirty={dirty:3d} {dict(c)}")
    out = f"{base}/pause_probe2.json"
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    print("saved:", out)
