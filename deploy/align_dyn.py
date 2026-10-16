#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同文案对齐下的「起伏对位」分析 —— 判断缺陷是不是引擎级（韵律/音色）

## 为什么做这个
前面两条假设（停顿下陷、局部顶起）在**正确 EL 基线**上都被证伪：
    句中空洞  我们 4.83 → 9.67 处/分   vs   EL 12.99
    孤立短音  我们 13.29 → 14.80 处/分  vs   EL 16.01
我们**比 EL 更少**。既然不是"多/深"，那剩下的可能就是"**位置不对**"——
起伏长在不该长的地方，听感就是"音量大小有变化 / 突然出来短音 / 少了点感情"。

## 做法
两个文件是同一份文案、同一秒同一句（已逐点 ASR 验证），所以可以直接**逐帧对齐**比：
1. 25ms 帧电平 E(t)；只保留两边都在发声（>−45dB）的帧。
2. 各自减去 2s 滑动中位 = 「本地起伏」D(t)（去掉音色/整体电平差）。
3. 算 D_我们 与 D_EL 的逐帧相关；再把各自的「顶起点」（D 的局部极大，突出度 ≥4dB）
   在时间上做**命中率**：我们的一个顶起，±300ms 内是否有 EL 的顶起。
   —— 命中率高 = 起伏长在 EL 也长的地方（只是量级差）；
      命中率低 = 起伏位置是随机的（这才是"不该有的起伏"）。

用法：venv/bin/python align_dyn.py
"""
import os
import subprocess

import numpy as np
import soundfile as sf

SR = 24000
FR = 0.025
HOP = 0.005
SPAN = 198.6
TMP = "/mnt/tts/diag/_dyn"
os.makedirs(TMP, exist_ok=True)
A = "/mnt/tts/diag/xf5272/prod_before_silfloor.mp3"
EL = "/mnt/tts/data/cases/52727ec43154/merged.wav"


def load(src, tag):
    o = os.path.join(TMP, "%s.wav" % tag)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-t", "%.2f" % SPAN,
                    "-i", src, "-ac", "1", "-ar", str(SR), o], check=True)
    return sf.read(o, dtype="float32")[0]


def env_db(y):
    h = int(HOP * SR)
    n = len(y) // h
    m = y[:n * h].astype(np.float64).reshape(n, h)
    return 20.0 * np.log10(np.sqrt((m ** 2).mean(axis=1)) + 1e-12)


def smooth_mov(x, win):
    """长度不变的滑动均值（edge padding）"""
    tot = win - 1
    pl, pr = tot // 2, tot - tot // 2
    xp = np.pad(x, (pl, pr), mode="edge")
    c = np.cumsum(np.insert(xp, 0, 0.0))
    return (c[win:] - c[:-win]) / win


def local_dyn(db, win_s=2.0):
    w = int(win_s / HOP)
    base = smooth_mov(db, w)                 # 2s 滑动均值 = 本地基准
    return db - base


# 允许 ±3 帧（±15ms）的微漂移：取 lag 最大相关的那个对齐
MAXLAG = 12


def best_lag(da, de, mask):
    best, bl = -2.0, 0
    for lag in range(-MAXLAG, MAXLAG + 1):
        x = np.roll(da, lag)[mask]
        y = de[mask]
        if x.size < 100:
            continue
        r = np.corrcoef(x, y)[0, 1]
        if r > best:
            best, bl = r, lag
    return best, bl


def peaks(d, mask, prom=4.0, mind=0.12):
    """顶起点：局部极大，突出度 ≥prom dB，间隔 ≥mind"""
    idx = []
    n = len(d)
    gap = int(mind / HOP)
    for i in range(1, n - 1):
        if not mask[i]:
            continue
        if d[i] >= d[i - 1] and d[i] > d[i + 1] and d[i] > prom:
            if not idx or i - idx[-1] >= gap:
                idx.append(i)
            elif d[i] > d[idx[-1]]:
                idx[-1] = i
    return np.array(idx) * HOP


a = load(A, "A")
e = load(EL, "EL")
da, de = env_db(a), env_db(e)
n = min(len(da), len(de))
da, de = da[:n], de[:n]
m = (da > -45) & (de > -45)
print("对比帧数 %d（两边都在发声的占 %.1f%%）" % (n, 100.0 * m.mean()))

Da, De = local_dyn(da), local_dyn(de)
r, lag = best_lag(Da, De, m)
print("\n【本地起伏的逐帧相关】 r = %+.3f（最优 lag = %d 帧 / %.0fms）"
      % (r, lag, lag * HOP * 1000))
print("  同一句话、同一个音节的「重音位置」若一致，r 应显著为正。")

De_s = np.roll(De, -lag)
mask2 = m & np.roll(m, -lag)

pa = peaks(Da, mask2)
pe = peaks(De_s, mask2)
print("\n【顶起点】我们 %d 个（%.2f 处/分）  EL %d 个（%.2f 处/分）"
      % (len(pa), len(pa) / (n * HOP) * 60, len(pe), len(pe) / (n * HOP) * 60))
hit = 0
tol = 0.30
for t in pa:
    if pe.size and np.min(np.abs(pe - t)) <= tol:
        hit += 1
print("  我们的顶起里，±%.0fms 内有 EL 顶起的比例 = %d/%d = %.1f%%"
      % (tol * 1000, hit, len(pa), 100.0 * hit / max(1, len(pa))))
e_hit = 0
for t in pe:
    if pa.size and np.min(np.abs(pa - t)) <= tol:
        e_hit += 1
print("  EL 的顶起里，±%.0fms 内有我们顶起的比例 = %d/%d = %.1f%%"
      % (tol * 1000, e_hit, len(pe), 100.0 * e_hit / max(1, len(pe))))

# 随机基线：把我们顶起的时间整体平移 5s 再算命中率
rnd = []
for sh in (3.7, 5.3, 7.1, 11.3, 17.9):
    sp = pa + sh
    sp = sp[sp < n * HOP]
    h = sum(1 for t in sp if pe.size and np.min(np.abs(pe - t)) <= tol)
    rnd.append(100.0 * h / max(1, len(sp)))
print("  随机基线（整体平移后）= %.1f%% ~ %.1f%%  ← 命中率若不显著高于它，起伏位置就是随机的"
      % (min(rnd), max(rnd)))

print("\n【本地起伏幅度分布】(dB)")
for nm, D in (("我们", Da), ("EL", De)):
    v = D[mask2]
    q = np.percentile(v, [1, 5, 50, 95, 99])
    print("  %-4s p1 %+6.1f  p5 %+6.1f  中位 %+6.1f  p95 %+6.1f  p99 %+6.1f  σ %.2f"
          % (nm, q[0], q[1], q[2], q[3], q[4], float(v.std())))
