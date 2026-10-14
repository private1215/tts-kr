#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pause_probe5.py — 「啊」声定位器：细粒度扫描所有间隙（含<0.15s短间隙），
在元音频段(150-500Hz基频区)找谐波凸起，区分：
  - rumble(<120Hz 电源嗡) → 不是「啊」
  - vowel(150-500Hz有峰+谐波) → 「啊」候选
对比 raw vs 处理后，判断是 TTS 真念了还是处理引入。
"""
import numpy as np, soundfile as sf, subprocess, re, json

BASE = "/mnt/tts/diag/cross-20261013"

def load16(path):
    import tempfile, os
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path, "-ar", "16000", "-ac", "1", t], check=True)
    y, sr = sf.read(t, dtype="float32")
    os.unlink(t)
    return y, sr

def find_gaps(path, thr=-40, mind=0.03):
    r = subprocess.run(["ffmpeg", "-nostats", "-i", path,
                        "-af", f"silencedetect=noise={thr}dB:d={mind}", "-f", "null", "-"],
                       capture_output=True, text=True)
    s = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r.stderr)]
    e = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    return [(a, e[i] if i < len(e) else None) for i, a in enumerate(s)]

def gap_profile(seg, sr):
    """返回 (max_above_db, spec_flat, peak_hz_in_voice, voice_band_ratio)"""
    fl10 = int(0.01 * sr)
    n = len(seg) // fl10
    if n < 3:
        return None
    fr = seg[:n * fl10].reshape(n, fl10)
    rms = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-12)
    med = np.median(rms)
    above = float(rms.max() - med)
    pk = int(np.argmax(rms))
    c0, c1 = max(0, pk * fl10 - int(0.03 * sr)), min(len(seg), pk * fl10 + int(0.03 * sr))
    bump = seg[c0:c1]
    if len(bump) < 64:
        return None
    w = np.hanning(len(bump))
    sp = np.abs(np.fft.rfft(bump * w)) ** 2
    freqs = np.fft.rfftfreq(len(bump), 1 / sr)
    gm = np.exp(np.mean(np.log(sp + 1e-12)))
    flat = float(gm / (np.mean(sp) + 1e-12))
    # 语音频段 150-500Hz 主峰
    band = (freqs >= 150) & (freqs <= 500)
    peak_hz = float(freqs[band][np.argmax(sp[band])]) if band.any() else 0.0
    # 语音频段能量占比
    vband = (freqs >= 120) & (freqs <= 4000)
    vr = float(sp[vband].sum() / (sp.sum() + 1e-12))
    return above, flat, peak_hz, vr

def scan(path, label):
    y, sr = load16(path)
    gaps = find_gaps(path)
    rows = []
    for (s, e) in gaps:
        if e is None or s < 1.0:
            continue
        i0, i1 = int(s * sr), int(e * sr)
        if i1 - i0 < int(0.03 * sr):
            continue
        seg = y[i0:i1]
        pr = gap_profile(seg, sr)
        if pr is None:
            continue
        above, flat, peak_hz, vr = pr
        kind = "clean"
        if above > 6 and flat < 0.02 and 120 <= peak_hz <= 600 and vr > 0.5:
            kind = "VOWEL?"      # 「啊」候选：谐波+基音频段峰+能量集中语音段
        elif above > 8 and flat < 0.02:
            kind = "harmonic?"   # 谐波但峰不在语音区
        elif above > 8:
            kind = "bump?"
        rows.append(dict(t=round(s, 2), dur=round(e - s, 3), above=round(above, 1),
                         flat=round(flat, 4), peak=round(peak_hz), vr=round(vr, 2), kind=kind))
    return label, rows

targets = [
    ("F_raw", f"{BASE}/X2_f.wav"),
    ("F_v2", f"{BASE}/X2f_v2.wav"),
    ("M_raw", f"{BASE}/X1_m.wav"),
    ("M_v4", f"{BASE}/X1m_v3.wav"),
]
allres = {}
for lab, p in targets:
    lab, rows = scan(p, lab)
    allres[lab] = rows
    from collections import Counter
    c = Counter(r["kind"] for r in rows)
    print(f"{lab:6s} gaps={len(rows):3d} {dict(c)}")
    vw = [r for r in rows if r["kind"] == "VOWEL?"]
    for r in vw[:25]:
        print(f"    VOWEL t={r['t']:7.2f} dur={r['dur']:.3f} above={r['above']:5.1f} flat={r['flat']:.4f} peak={r['peak']}Hz vr={r['vr']}")
json.dump(allres, open(f"{BASE}/pause_probe5.json", "w"), ensure_ascii=False, indent=1)

# 交叉对比
print()
print("=== 「啊」候选在 raw vs 处理后的位置匹配(±0.5s) ===")
for pair in (("F_raw", "F_v2"), ("M_raw", "M_v4")):
    a = [r for r in allres[pair[0]] if r["kind"] == "VOWEL?"]
    b = [r for r in allres[pair[1]] if r["kind"] == "VOWEL?"]
    both = [r for r in b if any(abs(r["t"] - x["t"]) < 0.5 for x in a)]
    print(f"{pair[0]}:{len(a)}处 {pair[1]}:{len(b)}处  同位置都有:{len(both)}  仅处理后新增:{len(b)-len(both)}")
