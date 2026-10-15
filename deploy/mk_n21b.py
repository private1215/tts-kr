#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重做第二十一轮（n）**第一题**的两个音频。

🔴 上一版的 bug（员工当场上报「B 选项没有音频」）：
    CLIPS 里 n_ctrl1 / n_ctrl2 / n_new **三条用同一个源、同一个起点 120.0**，
    ffmpeg 同参数编码 -> 三个文件 **md5 完全相同**（3e11387a...）。
    页面上 A、B 是字节级同一个文件，看着就像「B 没做成、直接复制了 A」。

修法（保住控制题语义，同时让 A/B 是两个真文件）：
  1. 控制题**换位置**到第 80 秒（第 2/3/4 题用 120 秒，第 5 题用 40 秒），
     素材不再和任何一题撞车。
  2. ctrl1 / ctrl2 用**同一段素材**分别独立编码（控制题语义：两段一样快 -> 选 C），
     但靠 metadata title 让两个成品字节不同 -> md5/etag 不同，是两个真文件。

🔴 两个老坑别再踩：
  · 临时目录用 /mnt/tts/diag 下的，别用 /tmp（sudo 与当前用户属主冲突）
  · volume=%gdB，不是 %gdBB（多一个 B 会被解析成 +18dB，切片全爆表）
"""
import os
import subprocess

FB_DIR = "/mnt/tts/data/tone-ab"
TMPD = "/mnt/tts/diag/_n21b"
TGT_LUFS = -26.1
DUR = 20.0

NEW = "/mnt/tts/diag/xf5272/prod_EL_male_8197d9ac8a35.wav"
T0 = 80.0          # 控制题专用位置：不与 120s(题2/3/4)、40s(题5) 重叠


def lufs(p):
    r = subprocess.run(["ffmpeg", "-i", p, "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    v = [l for l in r.stderr.splitlines() if "I:" in l and "LUFS" in l]
    return float(v[-1].split("I:")[1].split("LUFS")[0].strip()) if v else None


def main():
    os.makedirs(TMPD, exist_ok=True)
    src = os.path.join(TMPD, "ctrl.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "%.1f" % T0,
                    "-i", NEW, "-t", "%.1f" % DUR, src], check=True)

    gain = TGT_LUFS - lufs(src)
    print("控制题素材 %s 第 %.0fs 起 %.0fs，增益 %+.2fdB"
          % (os.path.basename(NEW), T0, DUR, gain))

    for name, title in (("n_ctrl1", "clip A"), ("n_ctrl2", "clip B")):
        mp3 = os.path.join(FB_DIR, "%s.mp3" % name)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                        "-af", "volume=%gdB" % gain,      # 🔴 dB 不是 %gdBB
                        "-b:a", "192k",
                        "-metadata", "title=%s" % title,
                        mp3], check=True)
        back = lufs(mp3)
        flag = "OK " if abs(back - TGT_LUFS) < 1.0 else "!! "
        print("%s%-9s LUFS %6.1f (目标 %.1f)" % (flag, name, back, TGT_LUFS))
    print("输出:", FB_DIR)


if __name__ == "__main__":
    main()
