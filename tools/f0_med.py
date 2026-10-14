# -*- coding: utf-8 -*-
"""量成品的 F0 中位（pyin / 16k，与 el_f0_fill.py 同口径），看有没有跑偏。

用法： python f0_med.py <wav> [wav...]
"""
import os
import sys

import librosa
import numpy as np

# 🔴 生产口径 vs 诊断口径不是一回事（2026-10-14 晚发现）：
#    fix_chain.f0_median 用的是 frame_length=**1024** + **前 60s**；
#    我这套工具默认 2048 + 全篇。两个数差好几 Hz，跨口径比会得出假结论。
#    判「要不要补音高」必须用生产的那把尺子，所以加 --prod 切换。
FL = 1024 if "--prod" in sys.argv else 2048
WIN = 60.0 if "--prod" in sys.argv else 0.0
TARGET = {"male": (194.3, 3.3), "female": (278.8, 6.05)}   # 清洗后 EL 基线（诊断口径）


def f0_median(path):
    y, sr = librosa.load(path, sr=16000, mono=True)
    if WIN and len(y) > int(WIN * sr):
        y = y[:int(WIN * sr)]
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr,
                                 frame_length=FL, hop_length=256)
    v = f0[voiced] if voiced is not None else f0
    v = np.array([x for x in (v if v is not None else []) if not np.isnan(x)])
    if len(v) < 100:
        return None
    return float(np.median(v))


def main():
    if len(sys.argv) < 2:
        sys.exit("用法: f0_med.py <wav> [wav...]")
    for p in sys.argv[1:]:
        if not os.path.exists(p):
            print("缺文件", p)
            continue
        med = f0_median(p)
        if med is None:
            print("%-52s 测不出" % os.path.basename(p))
            continue
        line = "%-52s F0=%7.1f" % (os.path.basename(p)[:52], med)
        for g, (m, s) in TARGET.items():
            z = (med - m) / s
            if abs(z) < 6:
                line += "   %s z=%+.2f" % (g[0], z)
        print(line)


if __name__ == "__main__":
    main()
