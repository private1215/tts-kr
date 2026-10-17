# -*- coding: utf-8 -*-
"""gapburst48.py —— 纯能量 VAD 版的「停顿里冒出来的短音」检测器。

## 为什么不用 ASR 词时间戳（pauseburp48.py 的做法）
对长篇（3~5 分钟）跑 faster-whisper 太慢，而且 EL 基线长篇跑 ASR 会引入
"whisper 对两个引擎的偏差"，口径不再中立。**纯能量 VAD 对 EL / 我们完全同口径**，
这才是能下结论的对照。

## 判据
1. 5ms 帧 RMS → dB
2. 发声 = db > (全篇 mean_db + VOICE_MARGIN)
3. 停顿 = 连续非发声 ≥ MIN_GAP 秒
4. 停顿内的突起 = 连续 ≥ (停顿底噪 + DB_ABOVE) 且时长 MIN~MAX
5. **voiced 判定**：该段 <1kHz 能量占比 ≥ VOICE_RATIO
   —— 用来排除 click / 宽带噪声（员工说的「嘎哈」是能听出元音的短音）

用法：python3 gapburst48.py <音频...> [--json out.json]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 0.005
VOICE_MARGIN = -12.0       # 高于全篇均值多少 dB 算发声
MIN_GAP = 0.15
MIN_BURST = 0.05
MAX_BURST = 0.50
DB_ABOVE = 10.0
VOICE_RATIO = 0.50         # <1kHz 能量占比 ≥ 此值才算 voiced


def load(p):
    if not p.endswith(".wav"):
        t = tempfile.mktemp(suffix=".wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                        "-ar", str(SR), "-ac", "1", t], check=True)
        p = t
    x, sr = sf.read(p, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def voiced_ratio(seg, sr):
    if len(seg) < 32:
        return 0.0
    w = np.hanning(len(seg))
    sp = np.abs(np.fft.rfft(seg * w)) ** 2
    fr = np.fft.rfftfreq(len(seg), 1.0 / sr)
    low = sp[fr < 1000].sum()
    tot = sp.sum() + 1e-12
    return float(low / tot)


def analyze(path):
    x, sr = load(path)
    dur = len(x) / sr
    n = int(sr * FRAME)
    m = (len(x) // n) * n
    fr = x[:m].reshape(-1, n)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-12))
    mean_db = float(db.mean())

    voiced_mask = db > (mean_db + VOICE_MARGIN)
    # 停顿段
    gaps, i = [], 0
    while i < len(voiced_mask):
        if not voiced_mask[i]:
            j = i
            while j + 1 < len(voiced_mask) and not voiced_mask[j + 1]:
                j += 1
            if (j - i + 1) * FRAME >= MIN_GAP:
                gaps.append((i, j))
            i = j + 1
        else:
            i += 1

    hits = []
    for a, b in gaps:
        seg_db = db[a:b + 1]
        floor = float(np.percentile(seg_db, 10))
        over = seg_db >= (floor + DB_ABOVE)
        k = 0
        while k < len(over):
            if over[k]:
                j = k
                while j + 1 < len(over) and over[j + 1]:
                    j += 1
                d = (j - k + 1) * FRAME
                if MIN_BURST <= d <= MAX_BURST:
                    t0 = a * FRAME + k * FRAME
                    t1 = a * FRAME + (j + 1) * FRAME
                    seg = x[int(t0 * sr):int(t1 * sr)]
                    vr = voiced_ratio(seg, sr)
                    peak = float(seg_db[k:j + 1].max())
                    hits.append({"t0": round(t0, 2), "t1": round(t1, 2),
                                 "dur": round(d, 2),
                                 "peak_db": round(peak, 1),
                                 "rel_full_db": round(peak - mean_db, 1),
                                 "above_floor_db": round(peak - floor, 1),
                                 "voiced": bool(vr >= VOICE_RATIO),
                                 "low_ratio": round(vr, 2)})
                k = j + 1
            else:
                k += 1

    vb = [h for h in hits if h["voiced"]]
    return {"file": os.path.basename(path), "dur": round(dur, 1),
            "n_gaps": len(gaps), "n_burst": len(hits), "n_voiced": len(vb),
            "per100s": round(len(vb) / dur * 100, 2) if dur > 0 else 0,
            "mean_db": round(mean_db, 1),
            "hits": vb[:40]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rows = []
    for p in a.wavs:
        r = analyze(p)
        rows.append(r)
        print("%-34s %6.1fs 停顿%3d  突起%3d  voiced%3d  "
              "密度 %5.2f 处/100s" % (r["file"], r["dur"], r["n_gaps"],
                                     r["n_burst"], r["n_voiced"], r["per100s"]))
    if rows:
        d = [r["per100s"] for r in rows]
        print("\n合计 %d 条  voiced 密度：中位 %.2f  均值 %.2f 处/100s"
              % (len(rows), float(np.median(d)), float(np.mean(d))))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        print("saved", a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
