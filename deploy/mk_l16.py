#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第十九轮（l）A/B 切片。

本轮要验的两件事，都是**数据说变好了但没有一个人听过**的改动：
  1. 女声 EQ（Σz² 86.2 -> 28）：机器说好，人耳认不认？
  2. 男声自适应高频（Σz² 12.6 -> 9.8）：机器说好，人耳认不认？

切片设计（沿用已定案的规则）：
  - 20s，取自 120s 处（避开开场，且两篇都够长）
  - **裸切不 loudnorm**，只做 volume 归一到 -26.1 LUFS
  - 🔴 写完必须回读 LUFS 复核：`volume=%gdBB` 会被 ffmpeg 解析成 **+18dB**
    （2026-10-14 踩过，7 个切片全爆表）。正确写法是 `volume=%gdB`。
"""
import json
import os
import subprocess
import sys

FB_DIR = "/mnt/tts/data/tone-ab"
# 🔴 音频就放在 FB_DIR 根下（路由是 /fb-audio/{name}），不是 audio/ 子目录
OUT = FB_DIR
TARGET_LUFS = -26.1
DUR = 20.0

# (切片名, 源 wav, 起点秒)
CLIPS = [
    ("l_ctrl1", "/mnt/tts/diag/xf5272/fem_witheq.wav", 120.0),
    ("l_ctrl2", "/mnt/tts/diag/xf5272/fem_witheq.wav", 120.0),
    ("l_f_new", "/mnt/tts/diag/xf5272/fem_witheq.wav", 120.0),
    ("l_f_old", "/mnt/tts/diag/xf5272/fem_noeq.wav", 120.0),
    ("l_f_el", "/mnt/tts/data/cases/52727ec43154/merged.wav", 120.0),
    ("l_m_new", "/mnt/tts/diag/x8197/ab_hf/adapt.wav", 120.0),
    ("l_m_old", "/mnt/tts/diag/x8197/ab_hf/fixed.wav", 120.0),
    ("l_m_el", "/mnt/tts/data/cases/8197d9ac8a35/merged.wav", 120.0),
]


def lufs(p):
    r = subprocess.run(["ffmpeg", "-nostats", "-i", p, "-af",
                        "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    for line in reversed(r.stderr.splitlines()):
        if "I:" in line and "LUFS" in line:
            try:
                return float(line.split("I:")[1].split("LUFS")[0].strip())
            except Exception:
                pass
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, src, start in CLIPS:
        if not os.path.exists(src):
            print("缺源 %s" % src)
            continue
        dst = os.path.join(OUT, name + ".mp3")
        tmp = "/tmp/_l_%s.wav" % name
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "%.1f" % start,
                        "-t", "%.1f" % DUR, "-i", src, "-ar", "24000",
                        "-ac", "1", tmp], check=True)
        cur = lufs(tmp)
        if cur is None:
            print("量不出 LUFS：%s" % name)
            continue
        gain = TARGET_LUFS - cur
        # 🔴 单位就是 dB，别多打一个 B
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp,
                        "-af", "volume=%gdB" % gain,
                        "-codec:a", "libmp3lame", "-b:a", "128k", dst],
                       check=True)
        back = lufs(dst)
        flag = "OK " if back is not None and abs(back - TARGET_LUFS) < 1.0 \
            else "!! "
        print("%s %-10s %6.1f -> %6.1f LUFS (目标 %.1f)"
              % (flag, name, cur, back if back else float("nan"),
                 TARGET_LUFS))
        os.unlink(tmp)


if __name__ == "__main__":
    main()
