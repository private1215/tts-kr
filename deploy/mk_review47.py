#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 z45 的 A/B 素材导出成本地可听的 wav（附复核页）。

口径与员工听到的**完全一致**：直接用 /mnt/tts/fb/z45_*.mp3（已归一到 −26.1 LUFS），
只是转回 wav，不改任何增益 —— 这样你听到的就是员工听到的那一份。
"""
import os
import subprocess

SR = 24000
OUT = "/mnt/tts/diag/_review47"
FB = "/mnt/tts/fb"
os.makedirs(OUT, exist_ok=True)

NAMES = [
    ("s1_t12", "z45_a_s1", "z45_b_s1"),
    ("s2_t115", "z45_a_s2", "z45_b_s2"),
    ("s3_t320", "z45_a_s3", "z45_b_s3"),
    ("el_t30", "z45_a_el", "z45_b_el"),
]


def to_wav(src, dst):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", dst],
                   check=True)


for tag, a, b in NAMES:
    to_wav(os.path.join(FB, "%s.mp3" % a),
           os.path.join(OUT, "%s_A_现役.mp3".replace(".mp3", ".wav") % tag))
    to_wav(os.path.join(FB, "%s.mp3" % b),
           os.path.join(OUT, "%s_B_候选.mp3".replace(".mp3", ".wav") % tag))
print("导出目录：" + OUT)
for f in sorted(os.listdir(OUT)):
    print("  %-34s %.2f MB" % (f, os.path.getsize(os.path.join(OUT, f)) / 1e6))
