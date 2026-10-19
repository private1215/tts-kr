# -*- coding: utf-8 -*-
"""tailclean53b2.py —— tailclean53b 的**保采样率**版（句末粘连「嗯」尾音）。

## 与 53b 的差别
53b 写死 `SR = 16000`，`sf.write(dst, y, SR)` 一律降到 16 kHz。
2026-10-19 实测：16 kHz 成品压 mp3 后音色 0.9331 → **0.6903（不同人）**
（对照：44.1k 下 wav→mp3 互比 0.9999，mp3 完全透明）。
→ **先降采样再压 mp3 = 白白毁音色**，交付链路里不能有这一步。

## 做法
检测判据**完全复用 53b**（`find_tail_clusters`，16k 口径，一个字不改），
只是把「替换成等长静音」这一步**落在原生采样率的波形上**，写出也用原生 sr。
命中位置以秒为单位 → 映射回原生 sr 是精确的，不存在对齐误差。

## 为什么必须清它（2026-10-19 当前链路实测，L3 全篇 559s）
    我们 句末粘连「嗯」 56 个 (6.01/分钟)
    EL               36 个 (3.89/分钟)   → 我们 1.55×
员工 10-17 的原话：「好比说完某个字后有个嗯」。
🔴 目标是**对齐 EL 密度（≈36），不是清零** —— 人声天然有微弱气声，
   清到 0 会变成「过度清洗」，反而更假（文档 §1.2 第三步）。

用法：python3 tailclean53b2.py in.wav out.wav [--dry]
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from tailclean53b import SR, find_tail_clusters, load16  # noqa: E402


def read_native(p):
    x, sr = sf.read(p, dtype="float32")
    if getattr(x, "ndim", 1) > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32), int(sr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--dry", action="store_true", help="只报告不写文件")
    a = ap.parse_args()

    y, sr = read_native(a.src)
    x = load16(a.src)                      # 检测仍在 16k（判据口径不变）
    tails = find_tail_clusters(x)
    dur = len(x) / SR
    print("%s  %.1fs @%dHz  句末粘连型嗯/颤簇 %d 个 (%.2f/分钟)"
          % (os.path.basename(a.src), dur, sr, len(tails),
             len(tails) / dur * 60), flush=True)
    for s, d, pk, zcr, low, gp, gn, nseg in tails:
        print("   t=%7.2fs %3.0fms 峰值%.0fdB zcr=%.3f 低频=%.2f "
              "前间隔%.0fms 后间隔%.0fms 段数=%d"
              % (s, d * 1000, pk, zcr, low, gp * 1000, gn * 1000, nseg),
              flush=True)

    if a.dry or not tails:
        if not tails and not a.dry:
            sf.write(a.dst, y, sr)
            print("无需处理，已原样写出（%d Hz）" % sr, flush=True)
        return 0

    z = y.copy()
    for s, d, *_ in tails:
        i0 = int(round(s * sr))
        i1 = int(round((s + d) * sr))
        if 0 <= i0 < i1 <= len(z):
            z[i0:i1] = 0.0
    sf.write(a.dst, z, sr)
    print("-> %s  已清理 %d 处（原生 %d Hz，总时长不变）"
          % (a.dst, len(tails), sr), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
