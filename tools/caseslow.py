# -*- coding: utf-8 -*-
"""案例库自身规律普查：慢变占比 / f0 基线 / 句级能量 sd 的分布。

## 为什么需要它

r25 用 5 个案例得出「头号病因 = 句间能量没有段落结构」：
我们的慢变占比 MA10 是 0.18~0.28，案例是 0.31~0.54，ac1 我们 −0.19~−0.22
vs 案例 +0.04~+0.17。

但 **5 个样本不足以断定"案例都这样"** —— 万一这 5 条恰好是起伏大的？
现在案例库涨到 24 条（女 11 / 男 13，且已核验各自同人），
必须普查一遍：如果 24 条的慢变占比都稳定在某个区间，
那「案例慢变、我们逐句抖」就是**确凿的系统性差异**，方案 1 才坐实。

顺带它还能回答：
- 女声案例够了（11 例 90 分钟），**女声多案例 EQ 联合拟合**能不能开
- 案例库变大后，响度中位 / PROSO A/E / 停顿样本 / 峰值中位的目标值各是多少

用法（服务器，qwen 环境）：
    venv/bin/python tools/caseslow.py
    venv/bin/python tools/caseslow.py --gender female
"""
import os
import sys
import json
import glob
import argparse

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "app"))
from app.prosody_report import sent_windows, f0_of     # noqa: E402
from tools.drift import load, SR, CASES_DIR            # noqa: E402
from tools.sentvar import slow_share                    # noqa: E402

# 已确认的真重复（MD5 完全一致），保留较早那条
DUP_DROP = {"c951ceb2c176", "653fff1fcca5"}


def ac1(v):
    a = v - v.mean()
    return float(np.sum(a[:-1] * a[1:]) / max(np.sum(a * a), 1e-12))


def measure(x, sr):
    segs, _, _ = sent_windows(x, sr)
    db, f0 = [], []
    for a, b in segs:
        s = x[a:b]
        if len(s) < int(0.15 * sr):
            continue
        db.append(20 * np.log10(np.sqrt(np.mean(s ** 2)) + 1e-12))
        try:
            f = f0_of(s, sr)
            if f is not None and len(f) >= 5:
                f0.append(float(np.median(f)))
        except Exception:
            pass
    db = np.asarray(db)
    f0 = np.asarray(f0)
    if len(db) < 10 or len(f0) < 10:
        return None
    return dict(
        n=len(db), e_sd=float(db.std()), e_ac1=ac1(db),
        e_slow=slow_share(db, (10,)).get(10, float("nan")),
        f0_med=float(f0.mean()), f0_sd=float(f0.std()),
        f0_slow=slow_share(f0, (10,)).get(10, float("nan")),
        f0_ac1=ac1(f0),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gender", default="")
    ap.add_argument("--out", default="/mnt/tts/diag/caseslow.json")
    a = ap.parse_args()

    meta = {}
    p = "/tmp/cases.json"
    if os.path.exists(p):
        for c in json.load(open(p)):
            meta[c["id"]] = c

    rows = []
    for d in sorted(glob.glob(os.path.join(CASES_DIR, "*", ""))):
        cid = os.path.basename(d.rstrip("/"))
        if cid in DUP_DROP:
            continue
        g = (meta.get(cid, {}).get("gender") or "unknown")
        if a.gender and g != a.gender:
            continue
        w = os.path.join(CASES_DIR, cid, "merged.wav")
        if not os.path.exists(w):
            continue
        x = load(w, SR)
        if x is None or len(x) < SR * 60:
            continue
        m = measure(x, SR)
        if not m:
            print(f"  {cid} 样本不足，跳过", flush=True)
            continue
        m["id"] = cid
        m["gender"] = g
        rows.append(m)
        print(f"  {cid} {g:<7} n={m['n']:4d}  能量 sd{m['e_sd']:5.2f} "
              f"ac1{m['e_ac1']:+.2f} 慢变{m['e_slow']:.2f}   "
              f"f0 {m['f0_med']:6.1f} sd{m['f0_sd']:5.2f} "
              f"慢变{m['f0_slow']:.2f}", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rows, open(a.out, "w"), ensure_ascii=False, indent=1)

    print("\n" + "=" * 92)
    print("案例库普查汇总（按性别）")
    print("=" * 92)
    for g in ("male", "female", "unknown"):
        r = [x for x in rows if x["gender"] == g]
        if not r:
            continue
        print(f"\n[{g}] {len(r)} 条")
        for k in ("e_sd", "e_ac1", "e_slow", "f0_med", "f0_sd", "f0_slow"):
            v = np.array([x[k] for x in r], dtype=np.float64)
            v = v[~np.isnan(v)]
            print(f"  {k:<9} 中位 {np.median(v):7.3f}   "
                  f"min {v.min():7.3f}  max {v.max():7.3f}  "
                  f"IQR {np.percentile(v,25):.3f}~{np.percentile(v,75):.3f}")
    print(f"\n已存 {a.out}")


if __name__ == "__main__":
    main()
