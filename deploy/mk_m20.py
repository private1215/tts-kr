#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第二十轮（m）女声切片：验证「房间底噪 -50dB -> -62dB」这一步改动。

为什么要问人
------------
机器侧已经很确定：端到端重跑，Σz² 34.0 -> 26.2、平坦度 +4.73σ -> +2.91σ。
但底噪声的是**停顿/间隙的听感**：
  · 太低 -> 回到 roomtone 当年要修的「数字真零 -> 过 lame 出电流声」；
  · 太高 -> 底噪被听见、频谱变平（flat 超标）。
机器量不出「间隙里有没有滋滋声」「停顿自不自然」，只能问耳朵。

🔴 出题铁规则（历年踩过的坑，别犯）：
  1. 每题只问一件事
  2. 同一维度下 **A/B 语义必须一致**（c 轮反过一次，全员选 A 被误读）
  3. 必须有一道隐藏控制题（同一段放两遍，正确答案=一样快）
  4. 音频放 `/mnt/tts/data/tone-ab/` **根目录**（路由 `/fb-audio/{name}`）
  5. `volume=%gdB` —— 多打一个 B 会被解析成 **+18dB**，写完必须回读 LUFS 复核
"""
import json
import os
import subprocess

FB_DIR = "/mnt/tts/data/tone-ab"
OUT = FB_DIR          # 🔴 根目录，不是 audio/ 子目录
TGT_LUFS = -26.1
DUR = 20.0
TMPD = "/mnt/tts/diag/_m20"      # 🔴 不用 /tmp：sudo 与当前用户对撞会 Permission denied

NEW = "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.wav"   # -62dB（新，已上线）
OLD = "/mnt/tts/diag/xf5272/prod_5272_rt50.wav"                # -50dB（旧）
EL = "/mnt/tts/data/cases/52727ec43154/merged.wav"             # EL 原文 5272

# (名字, 源文件, 起始秒)。底噪差异在停顿处最明显，所以切两个位置互相印证
CLIPS = [
    ("m_ctrl1", NEW, 120.0),
    ("m_ctrl2", NEW, 120.0),     # 与控制题同一段，正确答案=一样快
    ("m_rt_a",  NEW, 120.0),
    ("m_rt_b",  OLD, 120.0),
    ("m_rt_a2", NEW, 40.0),
    ("m_rt_b2", OLD, 40.0),
    ("m_el1",   EL,  120.0),
    ("m_el2",   EL,  40.0),
]


def lufs(p):
    r = subprocess.run(["ffmpeg", "-i", p, "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    v = [l for l in r.stderr.splitlines() if "I:" in l and "LUFS" in l]
    return float(v[-1].split("I:")[1].split("LUFS")[0].strip()) if v else None


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, src, t0 in CLIPS:
        mp3 = os.path.join(OUT, "%s.mp3" % name)
        os.makedirs(TMPD, exist_ok=True)     # 🔴 别用 /tmp：sudo 跑时残留文件属主冲突
        tmp = os.path.join(TMPD, "%s_norm.wav" % name)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "%.1f" % t0,
                        "-i", src, "-t", "%.1f" % DUR, tmp], check=True)
        cur = lufs(tmp)
        gain = TGT_LUFS - cur
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp,
                        "-af", "volume=%gdB" % gain,      # 🔴 dB，别写成 %gdBB
                        "-b:a", "192k", mp3], check=True)
        os.unlink(tmp)
        back = lufs(mp3)
        flag = "OK " if abs(back - TGT_LUFS) < 1.0 else "!! "
        print("%s%-10s LUFS %6.1f (目标 %.1f, 增益 %+.1fdB)  %.0f KB"
              % (flag, name, back, TGT_LUFS, gain,
                 os.path.getsize(mp3) / 1024))
    print("切片目录:", OUT)


if __name__ == "__main__":
    main()
