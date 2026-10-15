# -*- coding: utf-8 -*-
"""🔴 给 el_base_clean_*.json 补 `sd_robust` 字段（MAD×1.4826）。

## 为什么必须做（2026-10-15 实测发现）
`el_baseline_clean.py` 只剔了 2 条**显式**脏数据，
但**剩下 13 条里的离群值仍把 SD 撑大 9.6 ~ 26.8 倍**：

| 性别 | 维度 | 朴素 sd | 稳健 sd | 撑爆倍数 |
|---|---|---|---|---|
| 男 | 150~300 | 0.02390 | 0.00250 | **9.6x** |
| 男 | 500~1k  | 0.07826 | 0.00903 | **8.7x** |
| 女 | 60~150   | 0.00151 | 0.00006 | **26.8x** |
| 女 | 4k~8k    | 0.00423 | 0.00070 | **6.1x** |

判定门槛是 `|z| < 3`，而 `z = (我们 − mean) / sd`。
sd 被撑大 10 倍 → 同一份音频的 z 缩小 10 倍 → **「已达标」是假达标**。

实测反证：记忆里报「150~300 = −3.08σ（已压过头）」，
在稳健尺度下应是 `(0.187 − 0.2537) / 0.0025 = **−26.7σ**`，远超门槛。

## 附带产出：EL 自检分布（合格线）
除了稳健 sd，本脚本还会把「EL 标杆自己切片测出的 Σz²」统计出来，
写进 `el_selfcheck_<gender>.json`。
**EL 标杆自己就拿 239.8**（中位 59.6）→ 目标不是趋近 0，是落进这个区间。

## 用法
    python el_base_robust.py            # 只补 sd_robust
    python el_base_robust.py --selfcheck # 额外跑 EL 自检（慢，要切很多片段）

⚠️ 不改 mean/min/max/n，只**新增** sd_robust，旧脚本照常读 sd，不会被打断。
"""
import argparse
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = "/mnt/tts/diag/x8197/el_feats.json"
DIAG = "/mnt/tts/diag/x8197"
CASES = "/mnt/tts/data/cases"
SELF = "/mnt/tts/diag/selfcheck"
KEYS = ["f0_med", "f0_slope", "flat", "voiced", "b60_150", "b150_300",
        "b300_500", "b500_1000", "b1000_2000", "b2000_4000", "b4000_8000"]


def mad_sd(vals):
    """稳健标准差：MAD × 1.4826（对正态即等于 sd）。"""
    med = np.median(vals)
    return float(np.median(np.abs(vals - med)) * 1.4826)


def add_robust(gender):
    d = json.load(open(CACHE, encoding="utf-8"))
    base = json.load(open("%s/el_base_clean_%s.json" % (DIAG, gender),
                          encoding="utf-8"))
    rows = [v for v in d.values() if v.get("gender") == gender]
    print("=== %s  n=%d ===" % (gender, len(rows)))
    print("%-11s %12s %12s %10s %10s" %
          ("维度", "朴素sd", "稳健sd", "撑爆倍数", "稳健后Δ"))
    for k in KEYS:
        vals = np.array([r[k] for r in rows
                         if k in r and r[k] is not None], float)
        if len(vals) < 3 or k not in base:
            continue
        naive = base[k]["sd"]
        rob = mad_sd(vals)
        base[k]["sd_robust"] = rob
        ratio = naive / rob if rob > 1e-12 else float("inf")
        print("%-11s %12.6f %12.6f %9.1fx %10.2f%%"
              % (k, naive, rob, ratio, (naive - rob) / naive * 100))
    out = "%s/el_base_clean_%s.json" % (DIAG, gender)
    json.dump(base, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("-> 写入", out)


def run_selfcheck(gender, seg_len=20, fracs=(0.15, 0.50, 0.80)):
    """EL 标杆自己切片的 Σz² —— 这把尺子的噪声底。"""
    import sys
    sys.path.insert(0, HERE)
    import probe_g as PG
    base = PG.load_base(gender)
    keys = [k for k in PG.ORDER if k in base]
    d = json.load(open(CACHE, encoding="utf-8"))
    cases = [k for k, v in d.items() if v.get("gender") == gender]
    os.makedirs(SELF, exist_ok=True)
    scores, rows = [], []
    for cid in cases:
        src = "%s/%s/seg0.wav" % (CASES, cid)
        if not os.path.exists(src):
            src = "%s/%s/merged.wav" % (CASES, cid)
        if not os.path.exists(src):
            continue
        dur = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", src], capture_output=True, text=True).stdout.strip())
        for i, fr in enumerate(fracs):
            st = int(dur * fr)
            f = "%s/%s_%d.wav" % (SELF, cid, i)
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                            "-ss", str(st), "-t", str(seg_len),
                            "-ar", "16000", "-ac", "1", f], check=True)
            m = PG.measure(f, "full")
            zs = PG.zscored(m, base, keys, "robust")
            j = sum(v * v for v in zs.values())
            wk = max(zs, key=lambda k: abs(zs[k]))
            scores.append(j)
            rows.append({"case": cid, "start": st, "sigz2": round(j, 1),
                         "worst": wk, "worst_z": round(zs[wk], 2)})
            os.unlink(f)
    scores.sort()
    n = len(scores)
    res = {"gender": gender, "seg_len": seg_len, "n": n,
           "sigz2": [round(s, 1) for s in scores],
           "min": round(scores[0], 1) if n else None,
           "p25": round(scores[n // 4], 1) if n else None,
           "median": round(scores[n // 2], 1) if n else None,
           "p75": round(scores[3 * n // 4], 1) if n else None,
           "max": round(scores[-1], 1) if n else None,
           "rows": rows}
    out = "%s/el_selfcheck_%s.json" % (SELF, gender)
    json.dump(res, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n=== EL 自检(%s) n=%d ===" % (gender, n))
    print("Σz²  min=%.1f  p25=%.1f  中位=%.1f  p75=%.1f  max=%.1f"
          % (res["min"], res["p25"], res["median"], res["p75"], res["max"]))
    print("-> 写入", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--gender", default="both", choices=["male", "female", "both"])
    a = ap.parse_args()
    gs = ["male", "female"] if a.gender == "both" else [a.gender]
    for g in gs:
        add_robust(g)
        if a.selfcheck:
            run_selfcheck(g)