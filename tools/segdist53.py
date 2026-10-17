# -*- coding: utf-8 -*-
"""segdist53.py —— 全量发声段分布对比：找出「我们比 EL 多出来的那类段」。

## 为什么需要它

v1~v3 都是**先猜判据再验证**，三轮下来员工都说「清了一部分没清干净」：
    v1 (20~120ms, 静音>0.15)          我们男 9   EL 男 0   → 有效但只清 9 个
    v2/v3 (20~250ms, 静音>0.08, 浊音) 我们男 18  EL 男 16  → 一半可能是正常语音
    句末粘连鼻音                       我们男 21  EL 男 42  → 方向反了，封存

→ 不能再猜判据。**用数据找差异格子**：把所有发声段按
  (时长桶 × 浊音强度) 分格，对比我们 vs EL 的**每格密度（个/分钟）**，
  我们显著多的格子就是「嗯」的藏身处。

## 用法
    python3 segdist53.py <ours> <el> [--dump N]
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


def load16(p, sr=SR):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(sr), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def voiced_quality(a, sr=SR):
    if len(a) < 40:
        return 1.0, 0.0
    zcr = float(np.mean(np.abs(np.diff(np.sign(a)))) / 2)
    sp = np.abs(np.fft.rfft(a * np.hanning(len(a)))) ** 2
    fr = np.fft.rfftfreq(len(a), 1.0 / sr)
    low = float(sp[fr < 300].sum() / (sp.sum() + 1e-12))
    return zcr, low


def collect(p):
    x = load16(p)
    n = int(FRAME * SR)
    nfr = len(x) // n
    rms = np.sqrt((x[:nfr * n].reshape(nfr, n) ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-12)
    v = db > np.percentile(db, 35)
    segs, i = [], 0
    while i < len(v):
        if v[i]:
            j = i
            while j < len(v) and v[j]:
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1
    rec = []
    dur = len(x) / SR
    for k, (a, b) in enumerate(segs):
        s, d = a * FRAME, (b - a) * FRAME
        pre = (s - segs[k - 1][1] * FRAME) if k > 0 else 9.9
        nxt = (segs[k + 1][0] * FRAME - (s + d)) if k + 1 < len(segs) else 9.9
        seg = x[int(s * SR):int((s + d) * SR)]
        zcr, low = voiced_quality(seg)
        rec.append(dict(t=s, d=d * 1000, pre=pre * 1000, nxt=nxt * 1000,
                        pk=float(db[a:b].max()), zcr=zcr, low=low))
    return dur, rec


def density(rec, dur):
    """每格 (时长桶, 浊音档) 的密度（个/分钟）。"""
    buckets = [(20, 60), (60, 100), (100, 150), (150, 200), (200, 300),
               (300, 450), (450, 700), (700, 100000)]
    vlevels = [(0.00, 0.30), (0.30, 0.50), (0.50, 0.70), (0.70, 1.01)]
    grid = {}
    for bi, (lo, hi) in enumerate(buckets):
        for vi, (vlo, vhi) in enumerate(vlevels):
            grid[(bi, vi)] = []
    for r in rec:
        for bi, (lo, hi) in enumerate(buckets):
            if lo <= r["d"] < hi:
                break
        else:
            continue
        for vi, (vlo, vhi) in enumerate(vlevels):
            if vlo <= r["low"] < vhi:
                break
        else:
            continue
        grid[(bi, vi)].append(r)
    return grid, buckets, vlevels, dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ours")
    ap.add_argument("el")
    ap.add_argument("--dump", type=int, default=0)
    a = ap.parse_args()

    d1, r1 = collect(a.ours)
    d2, r2 = collect(a.el)
    g1, B, V, _ = density(r1, d1)
    g2, _, _, _ = density(r2, d2)

    print("=== %s (我们 %.1fs) vs %s (EL %.1fs) ==="
          % (os.path.basename(a.ours), d1, os.path.basename(a.el), d2))
    print("格子 = 时长桶 × 低频占比档（低频占比高=鼻浊音特征）")
    print()
    hdr = "%-14s" % "时长桶"
    for vlo, vhi in V:
        hdr += "%16s" % ("low %.2f-%.2f" % (vlo, vhi))
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for bi, (lo, hi) in enumerate(B):
        lab = "%d~%dms" % (lo, hi) if hi < 100000 else ">=%dms" % lo
        line = "%-14s" % lab
        for vi, (vlo, vhi) in enumerate(V):
            n1 = len(g1[(bi, vi)]) / d1 * 60
            n2 = len(g2[(bi, vi)]) / d2 * 60
            diff = n1 - n2
            rows.append((diff, bi, vi, n1, n2))
            mark = ""
            if diff > 0.4 and n1 > max(1.2 * n2, 0.5):
                mark = " *"
            line += "%12.2f/%.2f%s" % (n1, n2, " ")
        print(line)
    print()
    print("（每格格式：我们/EL，单位 个/分钟）")
    print()
    rows.sort(reverse=True)
    print("--- 我们比 EL 多的格子 Top6 ---")
    for diff, bi, vi, n1, n2 in rows[:6]:
        lo, hi = B[bi]
        vlo, vhi = V[vi]
        print("  时长 %d~%dms  低频%.2f~%.2f  我们%.2f vs EL%.2f  (+%.2f/分钟)"
              % (lo, hi, vlo, vhi, n1, n2, diff))
        if a.dump:
            for r in g1[(bi, vi)][:a.dump]:
                print("      t=%7.2fs %3.0fms 前静音%4.0fms 后静音%4.0fms "
                      "zcr=%.3f 低频=%.2f 峰值%.0fdB"
                      % (r["t"], r["d"], r["pre"], r["nxt"],
                         r["zcr"], r["low"], r["pk"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
