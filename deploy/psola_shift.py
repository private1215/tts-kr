#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PSOLA 变调：只改 F0，不动共振峰（替代 asetrate）。

## 为什么要换掉 asetrate
asetrate = 重采样，**F0 和共振峰一起搬**。实测（16k / Praat Burg）：
    EL 标杆        F0 187.9  F1 409.7  F2 1555.5
    VoxCPM2 裸片   F0 169.5  F1 406.2  F2 1549.8   <- 共振峰本来就对！
    asetrate 变调后 F0 188.3  F1 436.3  F2 1638.0   <- F2 被搬高 +5.3%
    PSOLA x1.10    F0 187.2  F1 423.1  F2 1558.3   <- F2 只偏 +0.2%
共振峰偏高 = 听感上「声道变短」= 感冒鼻塞 / 发假。
员工连着三轮都写「声音像感冒后发出的声音」，而频段能量占比早就调平了 ——
说明真凶是共振峰，不是能量分布。EQ 改不了共振峰中心频率，只能换变调方法。

## 方法
Praat Manipulation + PitchTier 缩放 + overlap-add 重合成（即 PSOLA）。
**不产生时长变化**，所以不需要 atempo 补偿（asetrate 那套三步里的第三步可省）。

## 用法
    python psola_shift.py raw.wav out.wav --ratio 1.10
    python psola_shift.py raw.wav out.wav --target-f0 194.3      # 自动算全局比例
    python psola_shift.py raw.wav out.wav --target-f0 194.3 --windowed
        # 分窗：每 60s 按自己的 F0 算比例（保留「分窗变调」抗漂移的定案）
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import parselmouth


def to16(path):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "16000", "-ac", "1", t], check=True)
    return t


def f0_median(snd, t0=None, t1=None):
    pit = snd.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=400)
    if t0 is None:
        t0, t1 = snd.xmin, snd.xmax
    ts = np.arange(t0 + 0.05, t1 - 0.05, 0.01)
    v = [pit.get_value_at_time(t) for t in ts]
    v = [x for x in v if x is not None and not np.isnan(x) and x > 0]
    return float(np.median(v)) if len(v) >= 20 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--ratio", type=float, default=None,
                    help="固定频率比（>1 升调）")
    ap.add_argument("--target-f0", type=float, default=None,
                    help="目标 F0 中位(Hz)，自动算比例")
    ap.add_argument("--windowed", action="store_true",
                    help="分窗变调：每窗按自己的 F0 算比例（抗长篇漂移）")
    ap.add_argument("--win", type=float, default=60.0)
    ap.add_argument("--sub", type=float, default=10.0,
                    help="分窗平滑时的细窗步长(秒)，越小台阶越细")
    a = ap.parse_args()

    if a.ratio is None and a.target_f0 is None:
        ap.error("需要 --ratio 或 --target-f0")

    tmp = to16(a.raw)
    snd = parselmouth.Sound(tmp)
    os.unlink(tmp)
    print("输入 %.1fs  F0=%.1f" % (snd.duration, f0_median(snd) or 0), flush=True)

    manip = parselmouth.praat.call(snd, "To Manipulation...", 0.01, 60, 400)
    pt = parselmouth.praat.call(manip, "Extract pitch tier")

    if a.ratio is not None:
        parselmouth.praat.call(pt, "Multiply frequencies...",
                               snd.xmin, snd.xmax, a.ratio)
        print("全局比例 %.4f" % a.ratio, flush=True)
    elif a.windowed:
        # 🔴 分窗必须用**插值后的连续比例**，不能每窗一个常数（2026-10-14 修）：
        #    常数比例会在窗边界留下音高台阶——实测相邻窗差最多 8%
        #    （窗3 x1.121 -> 窗4 x1.038），60s 处听得出来。
        #    asetrate 那套分窗当年就是靠 1:2:1 平滑解决的，PSOLA 同样要。
        #    做法：先按 --win 算出各窗比例，再在窗**中心**之间线性插值，
        #    用 --sub（默认 10s）的细窗逐个施加，台阶被切成听不出的小步。
        n = max(int(snd.duration / a.win), 1)
        win_r = []
        for i in range(n):
            t0 = snd.xmin + i * a.win
            t1 = min(t0 + a.win, snd.xmax)
            f = f0_median(snd, t0, t1)
            win_r.append(min(max(a.target_f0 / f, 0.90), 1.30)
                         if f and f > 0 else 1.0)
            print("  窗 %d [%.0f~%.0f] F0=%.1f -> x%.4f"
                  % (i, t0, t1, f or 0, win_r[-1]), flush=True)
        centers = [snd.xmin + (i + 0.5) * a.win for i in range(n)]

        def ratio_at(t):
            return float(np.interp(t, centers, win_r))

        step = a.sub
        t = snd.xmin
        while t < snd.xmax - 1e-6:
            t2 = min(t + step, snd.xmax)
            parselmouth.praat.call(pt, "Multiply frequencies...",
                                   t, t2, ratio_at((t + t2) / 2))
            t = t2
        print("分窗比例 %s -> 按 %.0fs 细窗插值施加（平滑）"
              % (["%.3f" % r for r in win_r], step), flush=True)
    else:
        f = f0_median(snd)
        r = min(max(a.target_f0 / f, 0.90), 1.30)
        parselmouth.praat.call(pt, "Multiply frequencies...",
                               snd.xmin, snd.xmax, r)
        print("全局 F0 %.1f -> 目标 %.1f  比例 %.4f"
              % (f, a.target_f0, r), flush=True)

    parselmouth.praat.call([pt, manip], "Replace pitch tier")
    out = parselmouth.praat.call(manip, "Get resynthesis (overlap-add)")
    out.save(a.out, "WAV")
    o = parselmouth.Sound(a.out)
    print("输出 %.1fs  F0=%.1f  -> %s" % (o.duration, f0_median(o) or 0, a.out))


if __name__ == "__main__":
    sys.exit(main())
