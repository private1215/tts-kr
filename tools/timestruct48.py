# -*- coding: utf-8 -*-
"""timestruct48.py —— 时序结构体检：我们 vs EL，**不需要时间对齐**。

## 为什么做这个
`tailburst48.py` 证明了「免对齐的统计」能在**时长不同、文案不同**的两条音频之间
做对照（裸 458s / 成品 453s / EL 459~525s 直接三方比）。这是本项目一直缺的能力：
以前所有对照都要求「同文案」，导致案例库 EL（36 条固定文案）和我们任意一条成品
**没法比**，只能靠员工盲听。

既然能量分布已经证明我们和 EL **几乎一致**（发声中位 −19.4 vs −19.0~−19.3），
差异只可能在**时序结构**。那就把时序结构全面量一遍。

## 指标（全部对时长/文案不敏感）
| 指标 | 含义 |
|---|---|
| voiced_pct | 发声帧占比（语速密度） |
| seg_med / seg_cv | 发声段时长中位数 / 变异系数（说话是否连贯） |
| gap_med / gap_cv | 静音段时长中位数 / 变异系数（停顿节奏） |
| seg_rate | 每秒发声段数（音节率） |
| long_gap_pct | 长静音(>=0.4s)占全部静音的比例 |

用法：python3 timestruct48.py <我们音频...> --el <EL 音频...>
"""
import argparse
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


def stats(path):
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
    segs, gaps, i = [], [], 0
    while i < len(vv):
        if vv[i]:
            j = i
            while j + 1 < len(vv) and vv[j + 1]:
                j += 1
            segs.append((j - i + 1) * FRAME)
            i = j + 1
        else:
            j = i
            while j + 1 < len(vv) and not vv[j + 1]:
                j += 1
            gaps.append((j - i + 1) * FRAME)
            i = j + 1
    sd = np.array(segs) if segs else np.array([0.0])
    gd = np.array(gaps) if gaps else np.array([0.0])
    return {"file": os.path.basename(path), "dur": round(dur, 1),
            "voiced_pct": round(float(voiced.mean() * 100), 1),
            "seg_med": round(float(np.median(sd)), 3),
            "seg_cv": round(float(sd.std() / max(sd.mean(), 1e-9)), 2),
            "gap_med": round(float(np.median(gd)), 3),
            "gap_cv": round(float(gd.std() / max(gd.mean(), 1e-9)), 2),
            "seg_rate": round(len(segs) / dur, 2) if dur else 0,
            "long_gap_pct": round(float((gd >= 0.4).mean() * 100), 1)}


KEYS = ["voiced_pct", "seg_med", "seg_cv", "gap_med", "gap_cv",
        "seg_rate", "long_gap_pct"]


def show(tag, rows):
    print("\n=== %s（%d 条）===" % (tag, len(rows)))
    print("  %-34s" % "素材" + "".join("%10s" % k for k in KEYS))
    for r in rows:
        print("  %-34s" % r["file"][:34] + "".join("%10s" % r[k] for k in KEYS))
    print("  %-34s" % "— 中位 —" + "".join(
        "%10.2f" % float(np.median([r[k] for r in rows])) for k in KEYS))
    return {k: float(np.median([r[k] for r in rows])) for k in KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", nargs="+", required=True)
    ap.add_argument("--el", nargs="+", required=True)
    a = ap.parse_args()
    o = show("我们", [stats(p) for p in a.ours])
    e = show("EL", [stats(p) for p in a.el])
    print("\n=== 差异（我们 − EL，标 * 的相对差 >20%）===")
    for k in KEYS:
        d = o[k] - e[k]
        rel = d / e[k] * 100 if e[k] else 0
        print("  %-14s %10.2f  vs %10.2f   %+8.2f  (%+6.1f%%)%s"
              % (k, o[k], e[k], d, rel, "  *" if abs(rel) > 20 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
