#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pause_clean_v3.py — 停顿清洗共用模块（第六轮反馈后重做）。

背景（2026-10-13 第六轮）：
- 旧策略「模板 = 全篇 RMS 最低的停顿中段」失败：RMS 最低 ≠ 干净，
  实测全篇扫不到 0 个合格 room tone 候选（<-50dB 且谱平坦>0.3 且稳定）。
  选出的模板本身带谐波凸起（女声 v2 残留 30 处、男声 v4 原始停顿 99/126 脏）。
- 新策略：不依赖原素材。用白噪声谱形整形合成干净 room tone
  （LF<120Hz 衰减除电源嗡、HF>5k 缓降），按每处停顿自身的中段 RMS
  （=现有底噪，保留不动）匹配电平，整体替换停顿内容，长度不变，
  边缘 5ms 交叉淡化保留自然过渡。只换内容不改长度 → 语速/节奏零影响。
"""
import numpy as np
import subprocess, re

def synth_tone(n, sr, rms, seed=42):
    """合成干净 room tone：白噪声 → LF<120Hz 余弦衰减、HF 5k→9k 衰减到 0.35。"""
    if n <= 0:
        return np.zeros(0, dtype="float32")
    rng = np.random.default_rng(seed)
    pad = 8192
    x = rng.standard_normal(n + pad)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    H = np.ones_like(f)
    lf = f < 120.0
    H[lf] = 0.5 * (1.0 - np.cos(np.pi * f[lf] / 120.0))
    hf = f > 5000.0
    H[hf] = 0.35 + 0.65 * 0.5 * (1.0 + np.cos(np.pi * np.clip((f[hf] - 5000.0) / 4000.0, 0, 1)))
    y = np.fft.irfft(X * H, n=len(x))
    y = y[pad // 2: pad // 2 + n]
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    cur = float(np.sqrt((y.astype(np.float64) ** 2).mean()))
    if cur > 0:
        y = y * (rms / cur)
    return y.astype("float32")

def gap_floor_rms(buf, sr, a, d):
    """停顿自身的中段 RMS（robust 底噪电平，不受凸起影响）。"""
    i0 = int((a + d * 0.3) * sr)
    i1 = int((a + d * 0.7) * sr)
    if i1 - i0 < 64:
        i0, i1 = int(a * sr), int((a + d) * sr)
    seg = buf[i0:i1]
    if len(seg) < 32:
        return None
    return float(np.sqrt((seg.astype(np.float64) ** 2).mean()))

def clean_gaps(buf, sr, gaps, min_dur=0.08, w_ms=5):
    """把 ≥min_dur 的停顿内容整体替换为合成干净 tone（长度不变）。
    电平 = 每处停顿自身中段 RMS（保留现有底噪）；边缘 w_ms 交叉淡化。
    返回 (buf, 清洗处数, 底噪电平列表)。"""
    w = max(1, int(w_ms * sr / 1000.0))
    ramp = np.linspace(0, 1, w)
    plan = sorted([(a, d) for a, d in gaps if d >= min_dur and a >= 0.0],
                  key=lambda x: -x[0])
    n_clean, floors = 0, []
    for a, d in plan:
        i0, i1 = int(a * sr), int((a + d) * sr)
        if i1 <= i0 or i1 > len(buf):
            continue
        if i1 - i0 < 2 * w + 32:
            continue
        target = gap_floor_rms(buf, sr, a, d)
        if target is None or target <= 0:
            continue
        floors.append(target)
        tone = synth_tone(i1 - i0, sr, target)
        orig_l = buf[i0:i0 + w].copy()
        orig_r = buf[i1 - w:i1].copy()
        buf[i0:i1] = tone
        buf[i0:i0 + w] = orig_l * (1 - ramp) + tone[:w] * ramp
        buf[i1 - w:i1] = orig_r * ramp[::-1] + tone[-w:] * ramp[::-1]
        n_clean += 1
    return buf, n_clean, floors

def scrub_stray(buf, sr, min_ms=20, max_ms=160, abs_thr=-48.0,
                rel_thr=12.0, sil_thr=-45.0, ctx_ms=80, pad_ms=15):
    """清除孤立人声碎片（第六轮「停顿处怪音」的真身）：
    TTS 在停顿处偶尔吐出 50-130ms 的语音电平元音碎片（-16~-23dB），
    起点在 -45dB 以下所以落在停顿区间内，但主体超出边界，停顿清洗切不掉。
    判定：10ms 帧 RMS > max(局部底噪+rel_thr, abs_thr) 的连续段（20-160ms），
    且其前后各 80ms 的中位电平都 < sil_thr（两侧都是静音=孤立）。
    → 替换为局部底噪电平的合成 tone（长度不变，5ms 交叉淡化）。
    正常语音不碰：词尾前接响、词头后延响，孤立性不成立。
    返回 (buf, 清除处数, 位置列表)。"""
    fl = int(0.01 * sr)
    n = len(buf) // fl
    if n < 40:
        return buf, 0, []
    fr = buf[:n * fl].reshape(n, fl)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-12)
    # 局部底噪：±0.5s 滑窗的 20 分位
    half = int(0.5 * sr / fl)
    floor = np.empty(n)
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        floor[i] = np.percentile(db[a:b], 20)
    loud = db > np.maximum(floor + rel_thr, abs_thr)
    # 连续段（允许跨 1 帧桥接）
    runs = []
    i = 0
    while i < n:
        if loud[i]:
            j = i
            gap = 0
            while j + 1 < n:
                if loud[j + 1]:
                    j += 1; gap = 0
                elif gap < 1 and j + 2 < n and loud[j + 2]:
                    j += 2; gap += 1
                else:
                    break
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    ctx = max(1, int(ctx_ms / 10))
    minf = max(2, int(min_ms / 10))
    maxf = max(minf, int(max_ms / 10))
    w = max(1, int(0.005 * sr))
    ramp = np.linspace(0, 1, w)
    hits = []
    for (a, b) in runs:
        durf = b - a + 1
        if durf < minf or durf > maxf:
            continue
        pre = db[max(0, a - ctx):a]
        post = db[b + 1:min(n, b + 1 + ctx)]
        pre_sil = (len(pre) == 0) or (np.median(pre) < sil_thr)
        post_sil = (len(post) == 0) or (np.median(post) < sil_thr)
        if not (pre_sil and post_sil):
            continue
        i0 = max(0, (a * fl) - int(pad_ms * sr / 1000))
        i1 = min(len(buf), ((b + 1) * fl) + int(pad_ms * sr / 1000))
        if i1 - i0 < 2 * w + 32:
            continue
        # 局部底噪电平（碎片前后静音段的中位）
        segs = []
        if a - ctx >= 0: segs.append(db[a - ctx:a])
        if b + 1 + ctx <= n: segs.append(db[b + 1:b + 1 + ctx])
        lvl = float(np.median(np.concatenate(segs))) if segs else -60.0
        tone = synth_tone(i1 - i0, sr, 10 ** (lvl / 20.0))
        orig_l = buf[i0:i0 + w].copy()
        orig_r = buf[i1 - w:i1].copy()
        buf[i0:i1] = tone
        buf[i0:i0 + w] = orig_l * (1 - ramp) + tone[:w] * ramp
        buf[i1 - w:i1] = orig_r * ramp[::-1] + tone[-w:] * ramp[::-1]
        hits.append((round(a * 0.01, 2), round(durf * 10, 0)))
    return buf, len(hits), hits

def detect_gaps_wav(path, thr=-45, mind=0.08):
    """对 wav 跑 silencedetect，返回 [(start, dur)]（剔除首尾 1s）。"""
    r = subprocess.run(["ffmpeg", "-nostats", "-i", path,
                        "-af", f"silencedetect=noise={thr}dB:d={mind}",
                        "-f", "null", "-"], capture_output=True, text=True)
    s = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r.stderr)]
    e = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    d = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                              "format=duration", "-of", "csv=p=0", path],
                             capture_output=True, text=True).stdout or 0)
    return [(a, b - a) for a, b in zip(s, e) if a > 1.0 and b < d - 1.0], d

def verify_pauses(path, label=""):
    """验收门：≥0.15s 停顿中，内部能量凸起 >8dB 的处数（probe2 同口径）。
    合成 tone 的凸起应≈0。"""
    import soundfile as sf
    data, fs = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    gaps, _ = detect_gaps_wav(path, thr=-45, mind=0.15)
    dirty = 0
    details = []
    for a, d in gaps:
        i0, i1 = int(a * fs), int((a + d) * fs)
        seg = data[i0:i1]
        fl = int(0.01 * fs)
        n = len(seg) // fl
        if n < 5:
            continue
        fr = seg[:n * fl].reshape(n, fl)
        rms = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-12)
        med = float(np.median(rms))
        above = float(rms.max()) - med
        if above > 8:
            dirty += 1
            details.append((round(a, 2), round(above, 1)))
    print(f"[verify]{label} 停顿={len(gaps)} 脏(凸起>8dB)={dirty} {details[:8]}")
    return len(gaps), dirty
