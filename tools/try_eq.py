# -*- coding: utf-8 -*-
"""EQ 变体对比：tone 收益 vs D/sil 代价，找平衡点（r25e）。

## 背景

r25e 用 tools/eqfit.py 拟合的新 EQ 上线后：

    tone  19.9 -> 8.76   ✅ +0.78 分
    D      4.68 -> 10.08 ❌ -0.43 分
    sil    1.90 -> 4.14  ❌ -0.16 分
    净 +0.27（离线只预测到 +0.78，因为离线只看了 tone）

## 为什么改 EQ 会动到 D 和 sil

D（语速CV）和 sil（静音占比）都依赖 **VAD 判句**，而验收的 VAD 是
**相对阈值** `帧电平 < p25 + 6 dB`。EQ 抬低频会连带把**停顿里的底噪**
（room tone / 编码噪声）一起抬上去 → 静音段电平上升 → p25 上升 →
阈值整体漂移 → 判句和停顿检出跟着变。

同一个坑 r24 已经踩过一次（软削波改能量分布上端 → E/F 崩）。

## 试什么

同一份响应矩阵（只测一次），用不同的**限幅策略**求解，然后**完整评分**
（不是只看 tone），看哪个净值最高：

* `full3`   —— 全部段 ±3 dB（= r25e 上线那版）
* `lo1`     —— 低频 3 段(65/92/131Hz) 限 ±1 dB，其余 ±3 dB
* `lo0`     —— 低频 3 段完全不动，其余 ±3 dB
* `maxdb15` —— 全部段 ±1.5 dB

用法（服务器 qwen 环境，完整评分要跑 ASR，比较慢）：
    venv/bin/python -u tools/try_eq.py --prefix r25d --only 1a6c20,00b6a4
"""
import os
import sys
import glob
import argparse
import subprocess

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.fidelity import (load, ltas_db, measure, score, EQ_BANDS,   # noqa
                            LTAS_FMAX)
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from eqfit import (PAIRS, N_BAND, SKIP_BAND, SR_EQ, SR_OUT, PROBE_DB,   # noqa
                   MIN_COL_RESP, concat_mp3, apply_gain, to_mp3, grid_of)

RT = "/mnt/tts/diag/eqwork"
CASES_DIR = os.environ.get("TTS_CASES_DIR", "/mnt/tts/data/cases")

# 低频前三段（65/92/131 Hz）：底噪主要在这里，抬它会把静音段一起抬起来
LOW_BANDS = [0, 1, 2]

VARIANTS = [
    ("base", None),                       # 不补偿（= r25d 现状）
    ("full3", dict(maxdb=3.0, lowdb=None)),
    ("lo1", dict(maxdb=3.0, lowdb=1.0)),
    ("lo0", dict(maxdb=3.0, lowdb=0.0)),
    ("maxdb15", dict(maxdb=1.5, lowdb=None)),
]


def solve(R, alive, target, cur, maxdb=3.0, lowdb=None, skip=None):
    """最小二乘求每段 dB；lowdb 给低频段单独限幅；skip 段强制不动。"""
    res = target - cur
    A = R[:, alive]
    x, *_ = np.linalg.lstsq(A, res, rcond=None)
    step = np.zeros(N_BAND)
    for k, j in enumerate(alive):
        if skip and j in skip:
            continue
        lim = lowdb if (lowdb is not None and j in LOW_BANDS) else maxdb
        step[j] = float(np.clip(x[k], -lim, lim))
    step[SKIP_BAND] = 0.0
    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="r25d")
    ap.add_argument("--only", default="1a6c20,00b6a4")
    ap.add_argument("--iters", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(RT, exist_ok=True)
    only = a.only.split(",")

    pairs = [p for p in PAIRS if p[1] in only]
    rows = {}
    for cid, short, gender in pairs:
        cw = os.path.join(CASES_DIR, cid, "merged.wav")
        fs = sorted(glob.glob(f"/mnt/tts/output/{a.prefix}-{short}/*.mp3"))
        if not os.path.exists(cw) or not fs:
            print(f"!! 跳过 {short}")
            continue
        print(f"\n=== {short} ({gender}) ===", flush=True)
        src = os.path.join(RT, f"src24_{short}.wav")
        concat_mp3(fs, src, SR_EQ)
        cm = measure(cw, "CASE")
        c_grid = grid_of(cw)

        # ---- 响应矩阵（每个案例各测一次，能量分布不同）------------------
        base_grid = grid_of(src)
        R = np.zeros((N_BAND, N_BAND))
        for j in range(N_BAND):
            if j == SKIP_BAND:
                continue
            g = np.ones(18)
            g[j] = 10.0 ** (PROBE_DB / 20.0)
            wav = os.path.join(RT, f"te_probe_{short}_{j}.wav")
            apply_gain(src, g, wav)
            R[:, j] = (grid_of(wav) - base_grid) / PROBE_DB
        alive = [j for j in range(N_BAND)
                 if np.nanmax(np.abs(R[:, j])) > MIN_COL_RESP]

        # ---- 当前（未补偿）的 grid，用于求残差 ---------------------------
        cur_grid = base_grid

        for name, cfg in VARIANTS:
            if cfg is None:
                gains = np.ones(18)
            else:
                gains = np.ones(18)
                db = np.zeros(18)
                for _ in range(a.iters):
                    st = solve(R, alive, c_grid, cur_grid,
                               maxdb=cfg["maxdb"], lowdb=cfg["lowdb"])
                    db[:N_BAND] += st
                    gains[:N_BAND] = 10.0 ** (np.clip(
                        db[:N_BAND], -cfg["maxdb"], cfg["maxdb"]) / 20.0)
                    # 重估当前 grid（wav 口径，快）
                    wav = os.path.join(RT, f"te_{short}_{name}.wav")
                    apply_gain(src, gains, wav)
                    cur_grid = grid_of(wav)
            tag = f"te_{short}_{name}"
            wav = os.path.join(RT, f"{tag}.wav")
            apply_gain(src, gains, wav)
            mp3 = os.path.join(RT, f"{tag}.mp3")
            to_mp3(wav, mp3)          # ⚠️ eqfit.to_mp3 不返回值
            pm = measure(mp3, "P")
            s, dv = score(cm, pm)
            rec = rows.setdefault(name, {})
            rec[short] = dict(score=s, devs=dv)
            print(f"  {name:<9} {s:6.2f}   "
                  f"tone {dv['tone']:5.1f}  D {dv['D']:5.1f}  "
                  f"sil {dv['sil']:5.1f}  C {dv['C']:5.1f}  "
                  f"E {dv['E']:5.1f}  F {dv['F']:5.1f}  A {dv['A']:5.1f}",
                  flush=True)

    keys = [s for _, s, _ in pairs if s in next(iter(rows.values()), {})]
    print("\n" + "=" * 78)
    print(f"{'变体':<10}{'均值':>8}" + "".join(f"{k:>10}" for k in keys))
    for name, _ in VARIANTS:
        if name not in rows:
            continue
        vs = [rows[name][k]["score"] for k in keys]
        print(f"{name:<10}{np.mean(vs):>8.2f}"
              + "".join(f"{v:>10.2f}" for v in vs))
    print("\n逐项均值对比（tone/D/sil/C/E/F/A）")
    for name, _ in VARIANTS:
        if name not in rows:
            continue
        line = "  %-9s" % name + "  ".join(
            f"{d}={np.mean([rows[name][k]['devs'][d] for k in keys]):5.1f}"
            for d in ("tone", "D", "sil", "C", "E", "F", "A"))
        print(line)


if __name__ == "__main__":
    main()
