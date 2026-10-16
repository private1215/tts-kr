#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""clickctx47.py —— 判断 click 落在「发声中」还是「停顿中」。

loc47 已证明：现役 omnivoice click 密度 0.48%，EL 0.26%，GS 0.01%。
下一步必须知道这些 click 是谁造的：

  · 落在**发声中**（高电平）→ 引擎本身 / 变调拉伸的时域伪影
  · 落在**停顿中或停顿边缘**（低电平）→ declick / silfloor 的段替换边界

判据：click 处前后各 50ms 的 RMS。
  · 若 max(左,右) < 局部中位 − 20dB  → 停顿中
  · 若 click 前后电平本来就差很多    → 边界（换段/拼接）
"""
import argparse
import numpy as np
import subprocess

SR = 16000


def load_wav(path, sr=SR):
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ar", str(sr),
         "-ac", "1", "-f", "f32le", "-"],
        capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32)


def db(v):
    return 20.0 * np.log10(max(v, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp3", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--thr", type=float, default=0.30)
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()

    x = load_wav(a.mp3)
    n20 = int(SR * 0.02)
    n50 = int(SR * 0.05)

    d = np.abs(np.diff(x))
    m = (len(d) // n20) * n20
    dmax = d[:m].reshape(-1, n20).max(axis=1)
    # 窗内最大跳变所在的样本位置
    arg = d[:m].reshape(-1, n20).argmax(axis=1)

    # 10ms RMS 用于判定电平
    n10 = int(SR * 0.010)
    m10 = (len(x) // n10) * n10
    r10 = np.sqrt((x[:m10].reshape(-1, n10) ** 2).mean(axis=1))
    med = np.median(r10)
    speech = r10 > (med * 10 ** (-20 / 20))     # 比中位低 20dB 以下算停顿

    hits = np.where(dmax >= a.thr)[0]
    print(f"\n===== {a.tag or a.mp3} =====")
    print(f"  |Δ|>={a.thr} 的窗 {len(hits)} 个 / {len(dmax)} "
          f"({len(hits)/len(dmax)*100:.2f}%)")
    print(f"  10ms RMS 中位 {db(med):.1f} dB")

    cat = {"发声中": 0, "停顿中": 0, "边界(左右电平悬殊)": 0}
    rows = []
    for i in hits[:a.top]:
        # click 样本位置
        si = i * n20 + arg[i]
        L = x[max(0, si - n50):si]
        R = x[si:si + n50]
        lrms = float(np.sqrt((L ** 2).mean())) if len(L) else 0.0
        rrms = float(np.sqrt((R ** 2).mean())) if len(R) else 0.0
        hi = max(lrms, rrms)
        if hi < med * 10 ** (-20 / 20):
            k = "停顿中"
        elif abs(db(lrms) - db(rrms)) > 20:
            k = "边界(左右电平悬殊)"
        else:
            k = "发声中"
        cat[k] += 1
        rows.append((si / SR, float(dmax[i]), db(lrms), db(rrms), k))

    # 全量分类
    cat_all = {"发声中": 0, "停顿中": 0, "边界(左右电平悬殊)": 0}
    for i in hits:
        si = i * n20 + arg[i]
        L = x[max(0, si - n50):si]
        R = x[si:si + n50]
        lrms = float(np.sqrt((L ** 2).mean())) if len(L) else 0.0
        rrms = float(np.sqrt((R ** 2).mean())) if len(R) else 0.0
        hi = max(lrms, rrms)
        if hi < med * 10 ** (-20 / 20):
            cat_all["停顿中"] += 1
        elif abs(db(lrms) - db(rrms)) > 20:
            cat_all["边界(左右电平悬殊)"] += 1
        else:
            cat_all["发声中"] += 1

    print("  全量分类：", "  ".join(f"{k}={v}" for k, v in cat_all.items()))
    print(f"\n  {'时间':>9}  {'|Δ|':>7}  {'左50ms':>8}  {'右50ms':>8}  类别")
    for t, v, l, r, k in rows:
        print(f"  {t:8.3f}s  {v:7.4f}  {l:8.1f}  {r:8.1f}  {k}")


if __name__ == "__main__":
    main()
