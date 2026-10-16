#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""量「停顿底噪」应该抬到多少：只看低于 -45dB 的帧，看它们的电平分布。

为什么要先量：如果直接照抄 EL 的 -52dB，而我们自己的房间底噪本来只在 -58dB，
硬抬到 -52dB 会在停顿里凭空多出 6dB 的嘶声 —— 又是一个"多了东西"的失败。
正确做法是让「最低谷」贴到 EL 的「最低谷」，同时不高于 EL 的「停顿常态」。

用法：venv/bin/python floor_lvl.py
"""
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

TMP = "/mnt/tts/diag/_floor"
os.makedirs(TMP, exist_ok=True)
FILES = [
    ("我们 女声成品", "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"),
    ("EL 5272 原声", "/mnt/tts/data/cases/5272" ),  # 占位，下面 main 里替换
]


def load(src, tag, sr=24000):
    o = os.path.join(TMP, "%s.wav" % tag)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ac", "1",
                    "-ar", str(sr), o], check=True)
    return sf.read(o, dtype="float32")[0]


def env_db(y, sr=24000, hop_s=0.005):
    hop = int(hop_s * sr)
    n = len(y) // hop
    e = np.sqrt(np.array([(y[i * hop:(i + 1) * hop] ** 2).mean() for i in range(n)]))
    return 20 * np.log10(e + 1e-12)


def rms(y):
    return float(np.sqrt((y ** 2).mean()) + 1e-12)


def hdr(name, y, sr=24000):
    db = env_db(y, sr)
    sp = db[db > np.percentile(db, 90)]            # 语音主体
    sp_db = 20 * np.log10(10 ** (sp / 20).mean() + 1e-12)
    lo = db[db < -45]                              # 停顿帧
    print("== %s  (%.1fs) ==" % (name, len(y) / sr))
    print("  语音主体 %6.1fdB   整体 RMS %6.1fdB" % (sp_db, 20 * np.log10(rms(y))))
    if len(lo) == 0:
        print("  无 <-45dB 帧")
        return
    q = np.percentile(lo, [0, 1, 5, 10, 25, 50, 75, 90, 99])
    print("  停顿帧 %.1f%%（%d 帧）分布：min %6.1f | p1 %6.1f | p5 %6.1f | p10 %6.1f "
          "| p25 %6.1f | 中位 %6.1f | p75 %6.1f | p90 %6.1f | p99 %6.1f"
          % (100.0 * len(lo) / len(db), len(lo), *q))
    print("  → 相对语音主体：min %+6.1fdB  中位 %+6.1fdB"
          % (q[0] - sp_db, q[5] - sp_db))
    # 深谷（连续 >=40ms 低于 -60dB）的最深值，这才是员工听到的那个"断"
    m = db < -60
    d = np.diff(m.astype(np.int8))
    st = list(np.where(d == 1)[0] + 1)
    en = list(np.where(d == -1)[0] + 1)
    if m[0]:
        st.insert(0, 0)
    if m[-1]:
        en.append(len(m))
    segs = [(s, e) for s, e in zip(st, en) if (e - s) * 0.005 >= 0.04]
    if segs:
        depth = [db[s:e].min() for s, e in segs]
        print("  深谷(<-60dB,>=40ms) %d 段：最深 %6.1fdB  中位 %6.1fdB  最浅 %6.1fdB"
              % (len(segs), min(depth), float(np.median(depth)), max(depth)))


def main():
    files = [
        ("我们 女声成品", "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"),
        ("EL 女声原声 52727ec", "/mnt/tts/data/cases/52727ec43154/merged.wav"),
        ("EL 男声原声 6cdd25b", "/mnt/tts/data/cases/6cdd25bc75f2/merged.wav"),
    ]
    for n, p in files:
        if p is None or not os.path.exists(p):
            print("== %s == 跳过（无文件）" % n)
            continue
        hdr(n, load(p, "f%d" % (abs(hash(n)) % 99999)))
        print()


if __name__ == "__main__":
    main()
