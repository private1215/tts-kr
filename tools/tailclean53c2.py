# -*- coding: utf-8 -*-
"""tailclean53c2.py —— tailclean53c 的**保采样率**版（句首紧贴型碎片）。

## 与 53c 的差别
53c 写死 `SR = 16000`，输出一律降到 16 kHz → 压 mp3 后音色崩到 0.6903（不同人）。
本版检测判据**完全复用 53c**（`find_head_clusters`，16k 口径不变），
只在原生采样率上做「置零」并原生 sr 写出。

## 为什么必须清它（2026-10-19 当前链路实测，L3 全篇 559s）
    我们 句首紧贴型碎片 15 个 (1.61/分钟)
    EL                2 个 (0.22/分钟)    → 我们 **7.3×**
这就是员工说的「**文案里没有的文字也会发音**」——它不是一个词，
而是发音起音前 20~50ms 的一个低能量鼻音碎片，听感像凭空冒出的「额/嗯」。
（ASR insert 实测为 0，whisper 转不出无意义鼻音，所以它不会出现在转写里，
  只能靠声学判据抓。）

🔴 目标对齐 EL（≈2 个/篇），不是清零。

用法：python3 tailclean53c2.py in.wav out.wav [--dry]
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from tailclean53c import SR, find_head_clusters, load16  # noqa: E402


def read_native(p):
    x, sr = sf.read(p, dtype="float32")
    if getattr(x, "ndim", 1) > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32), int(sr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?", default=None)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    y, sr = read_native(a.src)
    x = load16(a.src)
    hits = find_head_clusters(x)
    dur = len(x) / SR
    print("%s  %.1fs @%dHz  句首紧贴型碎片 %d 个 (%.2f/分钟)"
          % (os.path.basename(a.src), dur, sr, len(hits),
             len(hits) / dur * 60), flush=True)
    for s, d, pk, zcr, low, gp, gn in hits:
        print("   t=%7.2fs %3.0fms 峰值%.0fdB zcr=%.3f 低频=%.2f "
              "前间隔%.0fms 后间隔%.0fms"
              % (s, d * 1000, pk, zcr, low, gp * 1000, gn * 1000),
              flush=True)

    if a.dry or not hits:
        if a.dst and not hits:
            sf.write(a.dst, y, sr)
            print("无需处理，原样写出（%d Hz）" % sr, flush=True)
        return 0
    if not a.dst:
        print("未给输出路径，仅报告", flush=True)
        return 0

    z = y.copy()
    for s, d, *_ in hits:
        i0 = int(round(s * sr))
        i1 = int(round((s + d) * sr))
        if 0 <= i0 < i1 <= len(z):
            z[i0:i1] = 0.0
    sf.write(a.dst, z, sr)
    print("-> %s  已清理 %d 处（原生 %d Hz，总时长不变）"
          % (a.dst, len(hits), sr), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
