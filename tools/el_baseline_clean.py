# -*- coding: utf-8 -*-
"""从 el_feats.json（逐条缓存）产出**清洗后**的男女基线 JSON，供验收脚本共用。

## 为什么单独做
- el_baseline_male.json 是**原始**版（含脏数据 c3117b7b0fb1，f0 max 275.7），
  而 eq_search6.py 里那份清洗值是**手抄进代码**的，女声没有对应物。
- 女声基线此前根本没建（sigma 大得不可用的原因就是没剔 a3d39a095bac）。
- 现在统一：清洗规则写在一处，男女都出 JSON，脚本从 JSON 读，
  **不再把数字抄进代码**（抄进去就会腐烂，男声已经发生过一次）。

## 清洗规则
1. 显式脏数据：男 c3117b7b0fb1（标 male 但 F0=275.7、flat 落在女声簇）
                女 a3d39a095bac（8 个维度全部离群）
2. 同类兜底：男声 F0>230Hz 一律剔除；女声 F0<230Hz 一律剔除。
3. 迭代式 3-sigma 剔除（可关）：防止还有没被肉眼发现的离群案例。
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = "/mnt/tts/diag/x8197/el_feats.json"
KEYS = ["f0_med", "f0_range", "f0_slope", "flat", "voiced",
        "b60_150", "b150_300", "b300_500", "b500_1000",
        "b1000_2000", "b2000_4000", "b4000_8000"]

BAD = {"male": {"c3117b7b0fb1"}, "female": {"a3d39a095bac"}}
# 性别与基频明显不符的兜底阈值
F0_RANGE = {"male": (0, 230), "female": (230, 999)}


def load(gender, sigma_clip):
    cache = json.load(open(CACHE, encoding="utf-8"))
    rows = [(k, v) for k, v in cache.items()
            if (v.get("gender") or "") == gender]
    kept = []
    dropped = []
    for k, v in rows:
        if k in BAD.get(gender, set()):
            dropped.append((k, "显式脏数据"))
            continue
        f0 = v.get("f0_med")
        if f0:
            lo, hi = F0_RANGE[gender]
            if not (lo <= f0 < hi):
                dropped.append((k, "F0=%.1f 越界" % f0))
                continue
        kept.append((k, v))

    if sigma_clip > 0:
        for key in KEYS:
            vals = np.array([v.get(key) for _, v in kept
                             if v.get(key) is not None], dtype=float)
            if len(vals) < 5:
                continue
            mu, sd = vals.mean(), vals.std(ddof=1)
            if sd <= 1e-12:
                continue
            new = [(k, v) for k, v in kept
                   if v.get(key) is None
                   or abs(v[key] - mu) <= sigma_clip * sd]
            for k, _ in set(kept) - set(new):
                dropped.append((k, "%s 超 %s sigma" % (key, sigma_clip)))
            kept = new

    out = {}
    for key in KEYS:
        vals = np.array([v.get(key) for _, v in kept
                         if v.get(key) is not None], dtype=float)
        if len(vals) < 2:
            continue
        out[key] = {"mean": float(vals.mean()),
                    "sd": float(vals.std(ddof=1)),
                    "min": float(vals.min()), "max": float(vals.max()),
                    "n": int(len(vals))}
    return out, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-clip", type=float, default=0.0,
                    help=">0 时再做一轮迭代式 N-sigma 剔除（谨慎，默认关）")
    ap.add_argument("--out-dir", default="/mnt/tts/diag/x8197")
    a = ap.parse_args()

    for g in ("male", "female"):
        base, dropped = load(g, a.sigma_clip)
        n = max((v["n"] for v in base.values()), default=0)
        print("=" * 72)
        print("%s 清洗后基线  n=%d" % (g, n))
        print("=" * 72)
        for k in KEYS:
            if k in base:
                b = base[k]
                print("  %-11s %9.4f +- %.4f   (%9.4f ~ %9.4f)"
                      % (k, b["mean"], b["sd"], b["min"], b["max"]))
        if dropped:
            print("  剔除：%s" % ", ".join("%s(%s)" % d for d in dropped))
        p = os.path.join(a.out_dir, "el_base_clean_%s.json" % g)
        json.dump(base, open(p, "w"), ensure_ascii=False, indent=1)
        print("  -> %s\n" % p)


if __name__ == "__main__":
    main()
