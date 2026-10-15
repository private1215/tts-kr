#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第二十一轮（n）男声切片：验证「250Hz -0.45dB」这步改动。

背景：现役男声 150~300Hz = +3.96σ（8197、6cdd25 两篇一致 = 系统性），
与员工 j 轮 `m_cold` 选「我们像感冒鼻塞」精确对应。加 250Hz 衰减后端到端
Σz² 44.1 -> 23.1，各频段基本达标。

🔴 但机器说了不算：150~300 从 +3.96σ 变成 **-3.08σ**，也就是从「闷」变成了
   「偏薄」。这两头哪个更接近标杆、有没有引入新的不舒服，只有耳朵能判。

「原状」对照怎么来的：
    不重启服务，直接对当前成品施加 **+0.45dB @250Hz**（改动是可逆的 EQ，
    逆运算近似还原）。省一次重启，也保证两个片段来自**同一份生成**，
    不受 VoxCPM2 ±4σ 波动干扰（见 2026-10-14 记忆第六十九节）。

🔴 出题铁规则：
  · A/B 语义全轮一致（A 恒为新版）
  · 必须有一道隐藏控制题
  · 音频放 /mnt/tts/data/tone-ab/ **根目录**（路由 /fb-audio/{name}）
  · volume=%gdB，写完回读 LUFS 复核
"""
import os
import subprocess

FB_DIR = "/mnt/tts/data/tone-ab"
OUT = FB_DIR
TMPD = "/mnt/tts/diag/_n21"
TGT_LUFS = -26.1
DUR = 20.0

NEW = "/mnt/tts/diag/xf5272/prod_EL_male_8197d9ac8a35.wav"
EL = "/mnt/tts/data/cases/8197d9ac8a35/merged.wav"
OLD_SIM = os.path.join(TMPD, "old_sim.wav")   # 新版 + 250Hz +0.45dB ≈ 原状

CLIPS = [
    ("n_ctrl1", NEW, 120.0),
    ("n_ctrl2", NEW, 120.0),      # 与控制题同一段，正确答案=一样快
    ("n_new",   NEW, 120.0),
    ("n_old",   OLD_SIM, 120.0),
    ("n_el",    EL, 120.0),
    ("n_new2",  NEW, 40.0),       # 换位置再确认一次
    ("n_old2",  OLD_SIM, 40.0),
    ("n_el2",   EL, 40.0),
]


def lufs(p):
    r = subprocess.run(["ffmpeg", "-i", p, "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    v = [l for l in r.stderr.splitlines() if "I:" in l and "LUFS" in l]
    return float(v[-1].split("I:")[1].split("LUFS")[0].strip()) if v else None


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TMPD, exist_ok=True)
    # 逆向还原「原状」：把 -0.45dB 的衰减补回来
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", NEW,
                    "-af", "equalizer=f=250:t=q:w=1.1:g=0.45",
                    OLD_SIM], check=True)
    print("old_sim <- 新版 + 250Hz +0.45dB (近似原状)")

    for name, src, t0 in CLIPS:
        mp3 = os.path.join(OUT, "%s.mp3" % name)
        tmp = os.path.join(TMPD, "%s_norm.wav" % name)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "%.1f" % t0,
                        "-i", src, "-t", "%.1f" % DUR, tmp], check=True)
        gain = TGT_LUFS - lufs(tmp)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp,
                        "-af", "volume=%gdB" % gain,      # 🔴 dB，别写 %gdBB
                        "-b:a", "192k", mp3], check=True)
        os.unlink(tmp)
        back = lufs(mp3)
        flag = "OK " if abs(back - TGT_LUFS) < 1.0 else "!! "
        print("%s%-8s LUFS %6.1f (目标 %.1f, 增益 %+.1fdB)"
              % (flag, name, back, TGT_LUFS, gain))
    print("切片目录:", OUT)


if __name__ == "__main__":
    main()
