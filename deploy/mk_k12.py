#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第十四轮盲听切片（k12）+ 响度拉平。

为什么**两个位置**都切：v7 的修复是**分窗**的，第一窗变调比例 1.220、
后半段降到 1.045。开头 v5/v7 差别小（F0 196 vs 197），后半段差别最大
（v5 约 225Hz vs v7 198Hz）。只切开头会低估修复收益，只切后半段又看不出
「开头别扭」是否解决 —— 两个位置各一题，正好把两件事分开问。

🔴 响度用 volume 增益拉平，**不用 loudnorm**（会改动态，A/B 就不再是单变量）。
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
TARGET = -25.7

CUTS = [
    ("k12_v5_head", DIAG + "/m_v5.wav", 40, 35),
    ("k12_v7_head", DIAG + "/m_v7.wav", 40, 35),
    ("k12_v5_tail", DIAG + "/m_v5.wav", 300, 35),
    ("k12_v7_tail", DIAG + "/m_v7.wav", 300, 35),
    ("k12_el_tail", EL, 300, 35),
    ("k12_v7_mid", DIAG + "/m_v7.wav", 180, 35),
    ("k12_c1", DIAG + "/m_v7.wav", 300, 35),
    ("k12_c2", DIAG + "/m_v7.wav", 300, 35),
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
        rc = subprocess.run(["cp", tmp2, dst]).returncode
        if rc != 0:
            subprocess.run(["sudo", "-S", "-p", "", "cp", tmp2, dst],
                           input=_sudo_pw(), text=True, check=True)
        print("%-14s 源 %-46s LUFS %6.2f -> 增益 %+.2f dB -> %.2f"
              % (name, os.path.basename(src), cur, gain, lufs(dst)))
        os.unlink(tmpw)
        os.unlink(tmp2)


if __name__ == "__main__":
    main()
