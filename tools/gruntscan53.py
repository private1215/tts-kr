# -*- coding: utf-8 -*-
"""gruntscan53.py —— 用**绝对低门槛**重新分段后扫描「嗯」判据。

## 根本问题（为什么 v1~v4 一直漏）

`gruntclean53.py` 用 `db > np.percentile(db, 35)` 判发声段 —— 这是**相对**
门槛。员工说的「嗯」峰值只有 **-36~-44dB**（主语音 -15dB），
`gruntprobe53.py` 在标注位置直接看到了这些 5~30ms 的残片，
但它们**低于 p35，根本不成为「段」**，所以任何基于段列表的判据都抓不到。

→ 改用**绝对门槛**（-38 / -42 / -45 / -48 dB）分段，再扫描判据。

## 目标数量（由员工反馈反推）

    女声：70s 片段里听到 7 个 -> ~6/分钟 -> 全篇 460s 约 46 个
    男声：90s 片段里听到 4 个 -> ~2.7/分钟 -> 全篇 560s 约 25 个
    v2 判据只抓到 女 5 / 男 18 —— 女声漏了 10 倍

## 用法
    python3 gruntscan53.py <ours_m> <el_m> <ours_f> <el_f>
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


def build(p, vth):
    """用绝对门槛 vth 分段，返回 (segs, dur)。
    segs: [(start_s, dur_s, peak_db, zcr, low, pre_gap_s, nxt_gap_s)]
    """
    x = load16(p)
    n = int(FRAME * SR)
    nfr = len(x) // n
    rms = np.sqrt((x[:nfr * n].reshape(nfr, n) ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-12)
    v = db > vth
    raw, i = [], 0
    while i < len(v):
        if v[i]:
            j = i
            while j < len(v) and v[j]:
                j += 1
            raw.append((i, j))
            i = j
        else:
            i += 1
    segs = []
    for k, (a, b) in enumerate(raw):
        st = a * FRAME
        du = (b - a) * FRAME
        pk = float(db[a:b].max())
        seg = x[int(st * SR):int((st + du) * SR)]
        if len(seg) < 16:
            zcr, low = 1.0, 0.0
        else:
            zcr = float(np.mean(np.abs(np.diff(np.sign(seg)))) / 2)
            sp = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
            fr = np.fft.rfftfreq(len(seg), 1.0 / SR)
            low = float(sp[fr < 300].sum() / (sp.sum() + 1e-12))
        pre = (a - raw[k - 1][1]) * FRAME if k > 0 else 9.9
        nxt = (raw[k + 1][0] - b) * FRAME if k + 1 < len(raw) else 9.9
        segs.append((st, du, pk, zcr, low, pre, nxt))
    return segs, len(x) / SR


def cnt(segs, dur, vth_lo, max_ms, zcr_max, low_min, peak_db, gap_min):
    hi = max_ms / 1000.0
    n = 0
    for st, du, pk, zcr, low, pre, nxt in segs:
        if du > hi or du < 0.005:
            continue
        if pk > vth_lo:          # 峰值不能太高（太高是正常语音）
            continue
        if pk < peak_db:
            continue
        if not (zcr < zcr_max and low > low_min):
            continue
        if max(pre, nxt) < gap_min:   # 至少一侧要有间隙
            continue
        n += 1
    return n / dur * 60


def main():
    fs = sys.argv[1:5]
    if len(fs) < 4:
        print("需要 4 个文件：ours_m el_m ours_f el_f")
        return 1
    VTHS = [-38.0, -42.0, -45.0]
    MAXS = [120, 200, 300]
    ZCRS = [0.10, 0.14, 0.18]
    LOWS = [0.35, 0.50, 0.65]
    PEAKS = [-50.0, -44.0, -38.0]
    GAPS = [0.0, 0.05, 0.10]

    rows = []
    for vth in VTHS:
        B = [build(p, vth) for p in fs]
        print("vth=%.0f -> 段数 男%d/EL%d/女%d/EL%d"
              % (vth, len(B[0][0]), len(B[1][0]), len(B[2][0]), len(B[3][0])),
              flush=True)
        for mx, zc, lw, pk, gp in itertools.product(MAXS, ZCRS, LOWS, PEAKS, GAPS):
            om = cnt(B[0][0], B[0][1], vth + 3, mx, zc, lw, pk, gp)
            em = cnt(B[1][0], B[1][1], vth + 3, mx, zc, lw, pk, gp)
            of = cnt(B[2][0], B[2][1], vth + 3, mx, zc, lw, pk, gp)
            ef = cnt(B[3][0], B[3][1], vth + 3, mx, zc, lw, pk, gp)
            # 目标：男 ~2.7/分钟、女 ~6/分钟；且 EL 明显更少
            tgt = abs(om - 2.7) / 2.7 + abs(of - 6.0) / 6.0
            ratio = min((om + .05) / (em + .05), (of + .05) / (ef + .05))
            score = ratio - tgt
            rows.append((score, vth, mx, zc, lw, pk, gp, om, em, of, ef))

    rows.sort(reverse=True)
    print()
    print("Top20（score = 我们/EL比值 - 偏离目标量的惩罚）")
    print("%5s %5s %5s %5s %6s %5s | %7s %7s | %7s %7s"
          % ("vth", "maxms", "zcr<", "low>", "peak", "gap", "我们男", "EL男", "我们女", "EL女"))
    print("-" * 74)
    for sc, vth, mx, zc, lw, pk, gp, om, em, of, ef in rows[:20]:
        print("%5.0f %5d %5.2f %5.2f %6.0f %5.2f | %7.2f %7.2f | %7.2f %7.2f"
              % (vth, mx, zc, lw, pk, gp, om, em, of, ef))
    print()
    print("目标量：男 ~2.7/分钟（员工 90s 听 4 个）、女 ~6.0/分钟（70s 听 7 个）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
