# -*- coding: utf-8 -*-
"""tailburst48.py —— 「句末收干净了又冒出来」的归因统计（不需要时间对齐）。

## 为什么需要它
想回答「句末短音是**引擎原生**还是**后处理引入**」，最直接是拿同一条文案的
裸输出 vs 成品逐点比。但后处理有变速（453 vs 458s），**两条对不齐**，
逐点比会全错。

→ 改用**不需要对齐的统计**：看「长静音之前的 0.6s 窗口里有几个独立发声段」。

    正常：…最后一个词 ┃ 长静音 ┃ 下一句        → 窗口内 1 个发声段
    异常：…最后一个词 ┃ 短音 ┃ 长静音 ┃ 下一句  → 窗口内 2 个（或更多）

这个比值对时间轴平移完全不敏感，所以**裸输出 / 成品 / EL 可以直接三方对照**。

## 判据
- 5ms 帧 dB，`mean` = 全篇均值
- 发声帧：db > mean + VOICE_UP（默认 2 dB）
- 发声段：连续发声帧，间隙 < 0.05s 合并
- 长静音：连续非发声 ≥ GAP_MIN（默认 0.40s）
- 对每个长静音，数它起点前 WINDOW（0.60s）内的发声段个数

用法：python3 tailburst48.py <音频...> [--json out.json]
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
VOICE_UP = 2.0
MERGE = 0.05
GAP_MIN = 0.40
WINDOW = 0.60
DEPTH_DB = 30.0      # 间隙比语音低这么多 dB 才算"已经收干净了又冒出来"


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


def analyze(path):
    x, sr = load(path)
    dur = len(x) / sr
    n = int(sr * FRAME)
    m = (len(x) // n) * n
    rms = np.sqrt((x[:m].reshape(-1, n) ** 2).mean(axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-12))
    mean = float(db.mean())

    voiced = db > (mean + VOICE_UP)
    mg = int(MERGE / FRAME)
    vv = voiced.copy()
    i = 0
    while i < len(vv):
        if not vv[i]:
            j = i
            while j + 1 < len(vv) and not vv[j + 1]:
                j += 1
            if (j - i + 1) <= mg:
                vv[i:j + 1] = True
            i = j + 1
        else:
            i += 1

    segs, i = [], 0
    while i < len(vv):
        if vv[i]:
            j = i
            while j + 1 < len(vv) and vv[j + 1]:
                j += 1
            segs.append((i * FRAME, (j + 1) * FRAME))
            i = j + 1
        else:
            i += 1

    # 长静音
    gaps, i = [], 0
    while i < len(vv):
        if not vv[i]:
            j = i
            while j + 1 < len(vv) and not vv[j + 1]:
                j += 1
            t0, t1 = i * FRAME, (j + 1) * FRAME
            if t1 - t0 >= GAP_MIN:
                gaps.append((t0, t1))
            i = j + 1
        else:
            i += 1

    # 🔴 v2：不看"窗口内有几个段"（正常句末的最后几个音节也算多段，三方都 ~98%，
    #    没有区分度）。改看**长静音前最后两个发声段之间的间隙有多深**：
    #      正常：两个音节连着说，间隙只比语音低 ~10 dB
    #      异常：话已经收干净（-50 dB）又冒出来，间隙深 30 dB+
    depths = []
    for t0, _ in gaps:
        before = [(s, e) for s, e in segs if e <= t0 + 1e-6]
        if len(before) < 2:
            continue
        ps, pe = before[-2]          # 倒数第二个发声段
        ls, le = before[-1]          # 最后一个发声段（紧挨长静音）
        if ls <= pe:                 # 两段重叠/相接，无间隙
            continue
        ia, ib = int(pe / FRAME), int(ls / FRAME)
        if ib <= ia:
            ib = ia + 1
        gap_min = float(db[ia:ib].min())
        # 🔴 段可能不足 1 帧，切片会空 -> 必须钳位，否则 max() 抛
        #    "zero-size array to reduction operation maximum"
        def seg_max(a, b):
            i0 = int(a / FRAME)
            i1 = max(i0 + 1, int(b / FRAME))
            return float(db[i0:min(i1, len(db))].max())
        lvl = max(seg_max(ps, pe), seg_max(ls, le))
        depths.append(lvl - gap_min)
    d = np.array(depths) if depths else np.array([])
    deep = int((d >= DEPTH_DB).sum()) if len(d) else 0
    tot = len(d)
    return {"file": os.path.basename(path), "dur": round(dur, 1),
            "mean_db": round(mean, 1), "n_seg": len(segs),
            "n_gap": len(gaps), "n_deep": deep, "n_measured": tot,
            "deep_pct": round(deep / tot * 100, 1) if tot else 0.0,
            "median_depth": round(float(np.median(d)), 1) if tot else 0.0,
            "per100s": round(deep / dur * 100, 2) if dur > 0 else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rows = []
    for p in a.wavs:
        r = analyze(p)
        rows.append(r)
        print("%-34s %6.1fs 长静音%4d  间隙深度中位 %5.1f dB  深间隙(>=%ddB) %3d "
              "(%5.1f%%)  密度 %5.2f/100s"
              % (r["file"], r["dur"], r["n_gap"], r["median_depth"],
                 DEPTH_DB, r["n_deep"], r["deep_pct"], r["per100s"]))
    if rows:
        print("\n合计 %d 条：深度中位 %.1f dB  深间隙占比 %.1f%%  密度 %.2f 处/100s"
              % (len(rows),
                 float(np.median([r["median_depth"] for r in rows])),
                 float(np.mean([r["deep_pct"] for r in rows])),
                 float(np.mean([r["per100s"] for r in rows]))))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        print("saved", a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
