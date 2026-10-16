#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""迭代压制「局部顶起短音」直到收敛（一次性压制不收敛，实测）

## 为什么必须迭代（z43 实测证据）
z43 的 B 版是在成品上一次性压出来的，切片后重新检测：
    题1 90~110s：A 9 处 → B 3 处（95.5s 那个「额」确实消失了 ✅）
    题2 130~150s：A 9 处 → B 7 处（131.9s +19dB → 132.0s +9dB，只减轻）
    题3 160~180s：A 10 处 → B 7 处（170.5s +14dB → 170.9s +13dB，**几乎没变**）
两个原因：
 ① 压完一批后**相对关系变了**，原本不突出的位置反而顶起来
    （题1 里 95.5s 压掉后，隔壁 96.6s 冒出 +15dB）
 ② 相邻的两个顶起段**合并**成一个更长的段（96.4s+96.7s → 96.6s/335ms），
    按原判据的时长窗口重新判定，等于没压到位

## 做法
循环：检测 → 压制 → 重新检测，最多 MAX_ITER 轮，
直到顶起密度降到 EL 水平（EL 长篇 8.7 处/分钟，我们原始 25.1）。
🔴 不是"压得越狠越好"，是**收敛到 EL 的密度**就停（EL 本身也有 8.7 处）。

## 员工侧同时要改的（重要教训）
z43 用的 A/B 对比题，两段都有毛病时员工只能选「差不多 / 都不行」，
**拿不到"B 里到底还有没有"的信息**。下一轮必须改成 z9 那种
**逐段独立判断**（一段一题「这段有没有突然冒出的短音」），A、B 各问一遍。

用法：/mnt/tts/projects/qwen3-tts/venv/bin/python apply_bump2.py
"""
import importlib.util
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

SRC = "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"
TMP = "/mnt/tts/diag/_bump2"
os.makedirs(TMP, exist_ok=True)
EL_DENSITY = 8.7           # EL 长篇实测（处/分钟）
TARGET_DENSITY = 10.0      # 收敛目标


def _mod():
    spec = importlib.util.spec_from_file_location("ab", "/mnt/tts/diag/apply_bump.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["ab"] = m
    spec.loader.exec_module(m)
    return m


AB = _mod()


def to_wav(src, dst):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ar", "24000",
                    "-ac", "1", dst], check=True)
    return dst


def stats(path):
    y, sr = AB.load(path, "st")
    b, fl, vo, ns = AB.find_bumps(y, sr)
    dur = len(y) / sr
    return len(b), len(b) / (dur / 60.0), dur


def density(path):
    n, d, dur = stats(path)
    return n, d


def main():
    cur = os.path.join(TMP, "it0.wav")
    to_wav(SRC, cur)
    n, d = density(cur)
    print("起始：顶起 %d 处 = %.1f 处/分钟（EL 目标 %.1f）" % (n, d, EL_DENSITY))

    target_db = float(os.environ.get("TARGET_DB", "2"))
    max_iter = int(os.environ.get("MAX_ITER", "6"))
    hist = []
    for it in range(1, max_iter + 1):
        nxt_mp3 = os.path.join(TMP, "it%d.mp3" % it)
        nf, _ = AB.apply_bump(cur, nxt_mp3, bump_db=8.0, maxlen=0.35, target_db=target_db)
        # apply_bump 输出 mp3；转回 wav 继续迭代
        nxtw = os.path.join(TMP, "w%d.wav" % it)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", nxt_mp3, "-ar", "24000",
                        "-ac", "1", nxtw], check=True)
        n, d = density(nxtw)
        hist.append((it, nf, n, d))
        print("  第%d轮：压制 %3d 处 -> 剩余顶起 %3d 处 = %5.1f 处/分钟" % (it, nf, n, d))
        cur = nxtw
        if d <= TARGET_DENSITY:
            print("  ✅ 已收敛到目标（≤%.1f）" % TARGET_DENSITY)
            break
    if hist and hist[-1][3] > TARGET_DENSITY:
        print("  ⚠️ %d 轮后仍未到目标 %.1f（当前 %.1f）" % (max_iter, TARGET_DENSITY, hist[-1][3]))

    out = "/mnt/tts/diag/_bump2/bump2.mp3"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", cur, "-b:a", "192k", out], check=True)
    print("\n输出：%s" % out)
    print("BUMP2_DONE")


if __name__ == "__main__":
    main()
