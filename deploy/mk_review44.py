#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 z44 的 A/B 素材 + EL 同秒参照导出成本地可听的 wav（附复核页）。

口径与员工听到的完全一致：A/B 直接用 /mnt/tts/fb/z44_*.mp3（已归一到 −26.1 LUFS），
EL 参照按同一响度归一 —— 这样你听到的和员工听到的是同一份。
"""
import os
import shutil
import subprocess

TGT = -26.1
OUT = "/mnt/tts/diag/_review44"
os.makedirs(OUT, exist_ok=True)
FB = "/mnt/tts/fb"
EL = "/mnt/tts/data/cases/52727ec43154/merged.wav"

PAIRS = [
    ("p1_95s", "z44_a_mid.mp3", "z44_b_mid.mp3", 90.0),
    ("p2_131s", "z44_a_lo.mp3", "z44_b_lo.mp3", 130.0),
    ("p3_170s", "z44_a_tl.mp3", "z44_b_tl.mp3", 160.0),
]


def lufs(p):
    r = subprocess.run(["ffmpeg", "-i", p, "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    v = [l for l in r.stderr.splitlines() if "I:" in l and "LUFS" in l]
    return float(v[-1].split("I:")[1].split("LUFS")[0].strip()) if v else None


def to_wav(src, dst, t0=None, dur=20.0, gain=None):
    c = ["ffmpeg", "-y", "-v", "error"]
    if t0 is not None:
        c += ["-ss", "%.1f" % t0, "-t", "%.1f" % dur]
    c += ["-i", src, "-ac", "1", "-ar", "24000"]
    if gain is not None and abs(gain) > 1e-3:
        c += ["-af", "volume=%gdB" % gain]
    c += ["-c:a", "pcm_s16le", dst]
    subprocess.run(c, check=True)


for tag, a, b, t0 in PAIRS:
    for nm, f in (("A_修前", a), ("B_修后", b)):
        to_wav(os.path.join(FB, f), os.path.join(OUT, "%s_%s.wav" % (tag, nm)))
    ref = os.path.join(OUT, "%s_EL同秒参照.wav" % tag)
    to_wav(EL, ref, t0=t0, dur=20.0)
    g = TGT - lufs(ref)
    to_wav(EL, ref, t0=t0, dur=20.0, gain=g)
    print("%s  A=%.1f B=%.1f EL=%.1f LUFS"
          % (tag, lufs(os.path.join(OUT, "%s_A_修前.wav" % tag)),
             lufs(os.path.join(OUT, "%s_B_修后.wav" % tag)), lufs(ref)))

print("\n导出目录：" + OUT)
for f in sorted(os.listdir(OUT)):
    print("  %-30s %.2f MB" % (f, os.path.getsize(os.path.join(OUT, f)) / 1e6))
