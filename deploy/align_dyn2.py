#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""起伏「对位」分析 v2 —— 加了最关键的对照：**同引擎两次生成**的可复现性。

## 逻辑
A = 旧成品（生成1）  B = 修后新成品（生成2，同引擎/同音色/同文案）  EL = EL 原声
同一份文案、同一秒同一句（已逐点 ASR 验证），所以三者可以逐帧对齐比。

- r(A,B) = **同引擎可复现性天花板**。
  · 若 r(A,B) 高、r(A,EL) 低 → 我们自己的起伏是稳定的，只是和 EL 不一样
    （= 引擎级韵律差异，后处理救不了）。
  · 若 r(A,B) 也低 → 我们的重音位置**每次生成都在乱变**，
    这才是「音量大小有变化 / 突然出来短音 / 少了点感情」的直接来源。
- 顶起点的命中率同样做 A↔B 与 A↔EL 对照，并用「整体平移」当随机基线。
"""
import os
import subprocess

import numpy as np
import soundfile as sf

SR = 24000
HOP = 0.010                      # 10ms 帧，比 5ms 稳
FR = 0.050
SPAN = 198.6
TMP = "/mnt/tts/diag/_dyn"
os.makedirs(TMP, exist_ok=True)
FILES = [
    ("A 旧成品", "/mnt/tts/diag/xf5272/prod_before_silfloor.mp3"),
    ("B 修后新成品", "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"),
    ("EL 原声", "/mnt/tts/data/cases/52727ec43154/merged.wav"),
]


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
    tot = win - 1
    pl, pr = tot // 2, tot - tot // 2
    xp = np.pad(x, (pl, pr), mode="edge")
    c = np.cumsum(np.insert(xp, 0, 0.0))
    return (c[win:] - c[:-win]) / win


def local_dyn(db, win_s=2.0):
    return db - smooth_mov(db, int(win_s / HOP))


D = {}
E = {}
for nm, p in FILES:
    y = load(p, nm.replace(" ", ""))
    E[nm] = env_db(y)
    D[nm] = local_dyn(E[nm])
N = min(len(E[k]) for k in E)


def aligned_r(k1, k2, maxlag=12):
    d1, d2 = D[k1][:N], D[k2][:N]
    m = (E[k1][:N] > -45) & (E[k2][:N] > -45)
    best, bl = -2.0, 0
    for lag in range(-maxlag, maxlag + 1):
        r = np.corrcoef(np.roll(d1, lag)[m], d2[m])[0, 1]
        if r > best:
            best, bl = r, lag
    return best, bl


def peaks(d, mask, prom=8.0, mind_s=0.30):
    idx, gap = [], int(mind_s / HOP)
    for i in range(1, len(d) - 1):
        if mask[i] and d[i] >= d[i - 1] and d[i] > d[i + 1] and d[i] > prom:
            if not idx or i - idx[-1] >= gap:
                idx.append(i)
            elif d[i] > d[idx[-1]]:
                idx[-1] = i
    return np.array(idx) * HOP


def hit(p1, p2, tol=0.30):
    if len(p1) == 0:
        return 0.0
    return 100.0 * sum(1 for t in p1 if len(p2) and np.min(np.abs(p2 - t)) <= tol) / len(p1)


def rand_hit(p1, p2, tol=0.30):
    out = []
    for sh in (3.7, 5.3, 7.1, 11.3, 17.9, 23.1):
        sp = p1 + sh
        sp = sp[sp < N * HOP]
        out.append(hit(sp, p2, tol))
    return (min(out), max(out)) if out else (0, 0)


print("### 对比跨度：同一文案前 %.1fs ###\n" % SPAN)
ks = [k for k, _ in FILES]
print("【本地起伏逐帧相关 r】")
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        r, lag = aligned_r(ks[i], ks[j])
        print("  %-14s ↔ %-14s  r = %+.3f  (lag %+.0fms)"
              % (ks[i], ks[j], r, lag * HOP * 1000))

print("\n【顶起点（突出度 ≥8dB，最小间隔 0.30s）】")
P = {}
for k in ks:
    m = (E[k][:N] > -45)
    P[k] = peaks(D[k][:N], m)
    print("  %-14s %3d 个 = %.1f 处/分" % (k, len(P[k]), len(P[k]) / (N * HOP) * 60))

print("\n【顶起位置命中率（±300ms）】")
for i in range(len(ks)):
    for j in range(len(ks)):
        if i == j:
            continue
        lo, hi = rand_hit(P[ks[i]], P[ks[j]])
        print("  %-14s → %-14s  %.1f%%   （随机基线 %.1f~%.1f%%）"
              % (ks[i], ks[j], hit(P[ks[i]], P[ks[j]]), lo, hi))

print("\n【本地起伏幅度】")
for k in ks:
    v = D[k][:N][E[k][:N] > -45]
    q = np.percentile(v, [5, 50, 95])
    print("  %-14s p5 %+6.1f  中位 %+6.1f  p95 %+6.1f  σ %.2f" % (k, q[0], q[1], q[2], v.std()))
