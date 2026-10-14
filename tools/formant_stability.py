# -*- coding: utf-8 -*-
"""共振峰测量的**可重复性**测试：换 Praat 参数，看 F1/F2 抖多少。

## 为什么要测（2026-10-14）
混合链 R=1.045 量出 F1=438.9 / F2=1571.2，相对裸片(411/1495)是 +6.8% / +5.1%，
而理论上该只搬 +4.5%。两者不一致 -> 怀疑 F1 的**测量误差**就有 ~9Hz。
EL 的 F1 sigma 只有 5.1Hz，**误差比自然分布还大**，那 F1 的 z 就是噪声，
不能进目标函数（会把搜索带偏）。F2 sigma 8.6Hz 相对好一点，但也要确认。

## 判据
对**同一个文件**只改 Praat 参数（max_formant / 共振峰个数 / 分析窗），
若 F1 抖动 >> EL sigma(5.1) -> 不可信，FORMANT_W 只给 F2 或整项剔除。
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import parselmouth

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SETTINGS = [
    ("maxf5000 n5 ts0.01", 5000, 5, 0.01),
    ("maxf5500 n5 ts0.01", 5500, 5, 0.01),
    ("maxf4500 n5 ts0.01", 4500, 5, 0.01),
    ("maxf5000 n4 ts0.01", 5000, 4, 0.01),
    ("maxf5000 n5 ts0.02", 5000, 5, 0.02),
]


def seg16(path, start, length):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error",
                    "-ss", "%.2f" % start, "-t", "%.2f" % length,
                    "-i", path, "-ar", "16000", "-ac", "1", t], check=True)
    return t


def measure(path, maxf, nf, ts):
    pr = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True)
    dur = float(pr.stdout.strip() or 0)
    start, length = 0.10 * dur, min(180.0, max(5.0, dur - 0.10 * dur))
    t = seg16(path, start, length)
    s = parselmouth.Sound(t)
    os.unlink(t)
    pit = s.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=400)
    fm = s.to_formant_burg(time_step=ts, max_number_of_formants=nf,
                           maximum_formant=maxf)
    out = {}
    for tsel in (0.0, 0.5):
        # 同一段内的前后半段，看内容位置带来的抖动
        lo = start + tsel * (length / 2) - start
        hi = lo + length / 2
        ts_ = np.arange(max(0.05, lo + 0.05), min(s.duration - 0.05, hi), 0.01)
        F1, F2 = [], []
        for tt in ts_:
            f0 = pit.get_value_at_time(tt)
            if f0 is None or np.isnan(f0) or f0 <= 0:
                continue
            a, b = fm.get_value_at_time(1, tt), fm.get_value_at_time(2, tt)
            if a is None or b is None or np.isnan(a) or np.isnan(b) \
                    or a <= 0 or b <= 0:
                continue
            F1.append(a)
            F2.append(b)
        if len(F1) >= 50:
            out["half%d" % int(tsel * 2)] = (float(np.median(F1)),
                                             float(np.median(F2)))
    return out


def main():
    paths = sys.argv[1:] or ["m_hb_1.045.wav", "m_v10.wav"]
    for p in paths:
        p = p if os.path.isabs(p) else os.path.join(HERE, p)
        print("=" * 70)
        print(os.path.basename(p))
        all1, all2 = [], []
        for name, mf, nf, ts in SETTINGS:
            r = measure(p, mf, nf, ts)
            for half in sorted(r):
                f1, f2 = r[half]
                all1.append(f1)
                all2.append(f2)
                print("  %-18s half%s  F1=%7.1f  F2=%7.1f"
                      % (name, half, f1, f2))
        a1, a2 = np.array(all1), np.array(all2)
        print("  --> F1 极差 %.1f Hz (EL sigma 5.1)   标准差 %.1f"
              % (a1.max() - a1.min(), a1.std(ddof=1)))
        print("  --> F2 极差 %.1f Hz (EL sigma 8.6)   标准差 %.1f"
              % (a2.max() - a2.min(), a2.std(ddof=1)))


if __name__ == "__main__":
    main()
