#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第十五轮盲听切片（k13）+ 响度拉平。

🔴 与上一轮（k12）的两个关键改动：
1. **片段 35s → 20s**。上一轮 5 题里员工直接跳过 1 题、play 记录还缺失，
   疲劳是真实问题。感冒感/音高/语速这类**音色与韵律特征听 8~10 秒就够**，
   35 秒纯粹是负担。宁可让每道题都被答完，也不要题题 35 秒却只答一半。
2. **A/B 固定用「差别最大」的后半段（第 5 分钟处）**。分窗变调第一窗比例
   1.220、后半段降到 1.045 —— 开头 v5/v7 的 F0 只差 1Hz（196 vs 197，听不出来），
   后半段差 27Hz（225 vs 198）。拿开头做 A/B 等于让员工猜。
🔴 响度用 volume 拉平，**不用 loudnorm**（会改动态，A/B 就不是单变量了）。
   ⚠️ 写法必须是 `volume=%gdB`；多打一个 B（`dBB`）会被解析成 +18dB 增益。
"""
import os
import re
import subprocess
import tempfile

OUT = "/mnt/tts/data/tone-ab"
DIAG = "/mnt/tts/diag/x8197"
EL = "/mnt/tts/data/cases/8197d9ac8a35/merged.wav"
TARGET = -25.7
DUR = 20

CUTS = [
    ("k13_c1", DIAG + "/m_v7.wav", 300, DUR),
    ("k13_c2", DIAG + "/m_v7.wav", 300, DUR),
    ("k13_a_t", DIAG + "/m_v5.wav", 300, DUR),
    ("k13_b_t", DIAG + "/m_v7.wav", 300, DUR),
    ("k13_el_t", EL, 300, DUR),
    ("k13_b_h", DIAG + "/m_v7.wav", 120, DUR),
    ("k13_el_h", EL, 120, DUR),
    ("k13_b_m", DIAG + "/m_v7.wav", 180, DUR),
    ("k13_el_m", EL, 180, DUR),
    ("k13_b_s", DIAG + "/m_v7.wav", 420, DUR),
    ("k13_el_s", EL, 420, DUR),
    ("k13_b_q", DIAG + "/m_v7.wav", 240, DUR),
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
                           input="a19910119\n", text=True, check=True)
        got = lufs(dst)
        flag = "" if abs(got - (-26.1)) < 0.35 else "  ⚠️响度偏离，检查"
        print("%-10s 源 %-12s LUFS %6.2f -> 增益 %+.2f dB -> %.2f%s"
              % (name, os.path.basename(src), cur, gain, got, flag))
        os.unlink(tmpw)
        os.unlink(tmp2)


if __name__ == "__main__":
    main()
