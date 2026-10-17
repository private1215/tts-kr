# -*- coding: utf-8 -*-
"""gruntgrid53.py —— 判据网格扫描：找「我们多 / EL 少」且**不依赖前后静音**的判据。

## 为什么

z26→z28 三轮，员工都说「清了一部分没清干净」。根因是所有判据都要求
「前后静音 > 0.08s」，只能抓**孤立**的「嗯」；而员工说的是
「说完某个字后有个嗯」——**紧贴发言、中间没有静音**，永远抓不到。

但直接去掉静音条件会误伤（`gruntaudit53.py` D 判据：男 58 个里只有 13 个真嗯）。

→ **不再猜，做网格扫描**：枚举 (时长上限 × 静音门槛 × 浊音门槛 × 峰值门槛)，
  在「我们男 / EL男 / 我们女 / EL女」四个文件上分别计数，
  挑出「我们 >> EL」的组合。判据好坏用 **比值** 判定，不用绝对数。

用法：
    python3 gruntgrid53.py <ours_m> <el_m> <ours_f> <el_f>
"""
import itertools
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


def cache(p):
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
            segs.append((i, j, float(db[i:j].max())))
            i = j
        else:
            i += 1
    # 每段算 zcr / low
    feat = []
    for a, b, pk in segs:
        s = x[int(a * FRAME * SR):int(b * FRAME * SR)]
        if len(s) < 40:
            feat.append((1.0, 0.0))
            continue
        zcr = float(np.mean(np.abs(np.diff(np.sign(s)))) / 2)
        sp = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
        fr = np.fft.rfftfreq(len(s), 1.0 / SR)
        low = float(sp[fr < 300].sum() / (sp.sum() + 1e-12))
        feat.append((zcr, low))
    return dict(x=x, db=db, segs=segs, feat=feat, dur=len(x) / SR)


def count(c, max_ms, sil, zcr_max, low_min, peak_db):
    """按判据计数（个/分钟）。sil<=0 表示不要求前后静音。"""
    segs, feat, dur = c["segs"], c["feat"], c["dur"]
    hi = max_ms / 1000.0
    n = 0
    for k, (a, b, pk) in enumerate(segs):
        if k == 0 or k == len(segs) - 1:
            continue
        s = a * FRAME
        d = (b - a) * FRAME
        if d > hi or d < 0.020:
            continue
        if pk < peak_db:
            continue
        if sil > 0:
            pre = s - segs[k - 1][1] * FRAME
            nxt = segs[k + 1][0] * FRAME - (s + d)
            if pre <= sil or nxt <= sil:
                continue
        zcr, low = feat[k]
        if not (zcr < zcr_max and low > low_min):
            continue
        n += 1
    return n / dur * 60


def main():
    fs = sys.argv[1:5]
    if len(fs) < 4:
        print("需要 4 个文件：ours_m el_m ours_f el_f")
        return 1
    C = [cache(p) for p in fs]
    print("加载完成：")
    for p, c in zip(fs, C):
        print("  %-30s %.1fs  %d 段" % (os.path.basename(p), c["dur"], len(c["segs"])))
    print()

    MAXS = [150, 250, 350]
    SILS = [0.0, 0.05, 0.08]
    ZCRS = [0.10, 0.12, 0.16]
    LOWS = [0.35, 0.45, 0.55]
    PEAKS = [-36.0, -32.0]

    rows = []
    for mx, sil, zc, lw, pk in itertools.product(MAXS, SILS, ZCRS, LOWS, PEAKS):
        om = count(C[0], mx, sil, zc, lw, pk)
        em = count(C[1], mx, sil, zc, lw, pk)
        of = count(C[2], mx, sil, zc, lw, pk)
        ef = count(C[3], mx, sil, zc, lw, pk)
        # 比值（加平滑避免 /0）
        rm = (om + 0.05) / (em + 0.05)
        rf = (of + 0.05) / (ef + 0.05)
        # 要求：我们绝对量不能太少（否则清了也没意义），且 EL 少
        score = min(rm, rf) * min(om, 1.0)
        rows.append((score, mx, sil, zc, lw, pk, om, em, of, ef))

    rows.sort(reverse=True)
    print("判据网格 Top20（按 我们/EL 比值排序，要求男声女声同时占优）")
    print("%5s %5s %5s %5s %6s | %7s %7s | %7s %7s"
          % ("maxms", "sil", "zcr<", "low>", "peak", "我们男", "EL男", "我们女", "EL女"))
    print("-" * 66)
    for sc, mx, sil, zc, lw, pk, om, em, of, ef in rows[:20]:
        print("%5d %5.2f %5.2f %5.2f %6.0f | %7.2f %7.2f | %7.2f %7.2f"
              % (mx, sil, zc, lw, pk, om, em, of, ef))
    print()
    print("参考：现役 v3 判据 (250ms, sil0.08, zcr0.12, low0.45, peak-32)")
    om = count(C[0], 250, 0.08, 0.12, 0.45, -32)
    em = count(C[1], 250, 0.08, 0.12, 0.45, -32)
    of = count(C[2], 250, 0.08, 0.12, 0.45, -32)
    ef = count(C[3], 250, 0.08, 0.12, 0.45, -32)
    print("   我们男 %.2f vs EL男 %.2f  |  我们女 %.2f vs EL女 %.2f" % (om, em, of, ef))
    return 0


if __name__ == "__main__":
    sys.exit(main())
