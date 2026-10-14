#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第十七轮盲听切片（k14）：验证「EL 统计基线驱动的自动 EQ」是否真的更好。

与上一轮（k13）的差别只有一处变量：**EQ**
    v7 = 人工手调三段 EQ（lowshelf800 -1 / highshelf1500 +1.5 / highshelf4000 -6.5）
    v9 = eq_search6 以「清洗后 EL 基线的 z 距离最小」自动搜出来的三段 peaking
         (90Hz +7.5 / 350Hz +3 / 1400Hz +3)
其余全部相同（同一条 VoxCPM2 裸推理 → 同一条分窗变调 → 同一篇文案 x8197），
所以 A/B 是单变量对比。

频段改善（清洗后男声基线 n=13）：
    v7: 4-8k -6.83  2-4k +5.17  150-300 +3.76  60-150 -3.25  300-500 -2.58  (5 项超 2σ)
    v9: 全部 |z| <= 1.42                                                  (0 项超 2σ)
    flat(16k 口径) 0.0330 -> 0.0372（EL 0.0424），同步改善

🔴 沿用 k13 的两条定案：
1. 片段 20s、取第 300s 处（后半段是分窗变调差别最大的位置，开头 v5/v7 F0 只差 1Hz
   听不出来，后半段差 27Hz）。
2. 响度用 volume 拉平，**不用 loudnorm**（会改动态，A/B 就不是单变量了）。
   ⚠️ 写法必须是 `volume=%gdB`；多打一个 B（`dBB`）会被解析成 +18dB 增益。
"""
import os
import re
import subprocess
import tempfile

def _sudo_pw():
    """取 sudo 密码。密码绝不写进代码/注释（曾硬编码并推上 GitHub，已改用环境变量）。
    优先用环境变量 TTS_SUDO_PW；目标目录若已可写则根本不需要 sudo。"""
    pw = os.environ.get("TTS_SUDO_PW", "")
    if not pw:
        raise SystemExit("需要 TTS_SUDO_PW 环境变量（或把目标目录改成可写，就不必 sudo）")
    return pw + "\n"


OUT = "/mnt/tts/data/tone-ab"
DIAG = "/mnt/tts/diag/x8197"
EL = "/mnt/tts/data/cases/8197d9ac8a35/merged.wav"
V9 = DIAG + "/m_v10.wav"
V7 = DIAG + "/m_v7.wav"
TARGET = -25.7
DUR = 20

# 题 2 与题 3 用同一对音频、顺序相反 —— 用来检出「位置惯性」
CUTS = [
    ("k14_c1", V9, 300, DUR),      # 控制题：第 1 遍
    ("k14_c2", V9, 300, DUR),      # 控制题：同一段第 2 遍（正确答案=一样快）
    ("k14_n_t", V9, 300, DUR),     # 新 EQ
    ("k14_el_t", EL, 300, DUR),    # EL 标杆
    ("k14_o_t", V7, 300, DUR),     # 旧 EQ
    ("k14_n_m", V9, 180, DUR),     # 新 EQ（另一处，防员工只认一个位置）
    ("k14_el_m", EL, 180, DUR),    # EL 标杆（另一处）
]


def lufs(p):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", p, "-af", "ebur128",
                        "-f", "null", "-"], capture_output=True, text=True)
    ms = re.findall(r"I:\s+(-?[\d.]+)\s+LUFS", r.stderr)
    return float(ms[-1]) if ms else None


def main():
    for name, src, ss, dur in CUTS:
        assert os.path.exists(src), src
        tmpw = tempfile.mktemp(suffix=".wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(ss),
                        "-t", str(dur), "-i", src, "-ar", "44100", "-ac", "1",
                        tmpw], check=True)
        cur = lufs(tmpw)
        gain = round(TARGET - cur, 2)
        tmp2 = tempfile.mktemp(suffix=".mp3")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmpw,
                        "-af", "volume=%gdB" % gain,
                        "-codec:a", "libmp3lame", "-b:a", "128k",
                        "-ar", "44100", tmp2], check=True)
        dst = os.path.join(OUT, name + ".mp3")
        if subprocess.run(["cp", tmp2, dst]).returncode != 0:
            subprocess.run(["sudo", "-S", "-p", "", "cp", tmp2, dst],
                           input=_sudo_pw(), text=True, check=True)
        got = lufs(dst)
        flag = "" if abs(got - (-26.1)) < 0.35 else "  ⚠️响度偏离，检查"
        print("%-10s 源 %-12s LUFS %6.2f -> 增益 %+.2f dB -> %.2f%s"
              % (name, os.path.basename(src), cur, gain, got, flag))
        os.unlink(tmpw)
        os.unlink(tmp2)


if __name__ == "__main__":
    main()
