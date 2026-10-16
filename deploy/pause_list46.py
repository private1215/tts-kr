#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""长停顿清单：我们 vs EL，同文案同跨度。顺带看男声是不是同一个毛病。

背景：z46 的分桶表显示 EL 有 6 处 0.5~1.0s 的停顿，我们**一处都没有**。
「断句有异声」这句话的字面意思就是断句（停顿位置/长度）不一样，所以这条要钉死。
"""
import os
import subprocess

import numpy as np
import soundfile as sf

SR = 24000
HOP = 0.010
TMP = "/mnt/tts/diag/_plist"
os.makedirs(TMP, exist_ok=True)
PAIRS = [
    ("女声", 198.6, [
        ("我们·修前", "/mnt/tts/diag/xf5272/prod_before_silfloor.mp3"),
        ("我们·修后", "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"),
        ("EL 女声", "/mnt/tts/data/cases/52727ec43154/merged.wav"),
    ]),
    ("男声", 494.0, [
        ("我们·男声成品", "/mnt/tts/output/longprobe-z37-6cdd/longprobe-z37-6cdd-1.mp3"),
        ("EL 男声", "/mnt/tts/data/cases/6cdd25bc75f2/merged.wav"),
    ]),
]


def load(src, tag, dur):
    o = os.path.join(TMP, "%s.wav" % tag)
    if not os.path.exists(o):
        c = ["ffmpeg", "-y", "-v", "error", "-t", "%.1f" % dur, "-i", src,
             "-ac", "1", "-ar", str(SR), o]
        r = subprocess.run(c, capture_output=True)
        if r.returncode != 0 or not os.path.exists(o):
            return None
    return sf.read(o, dtype="float32")[0]


def env_db(y):
    h = int(HOP * SR)
    n = len(y) // h
    m = y[:n * h].astype(np.float64).reshape(n, h)
    return 20.0 * np.log10(np.sqrt((m ** 2).mean(axis=1)) + 1e-12)


def runs(mask):
    d = np.diff(mask.astype(np.int8))
    st = list(np.where(d == 1)[0] + 1)
    en = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        st.insert(0, 0)
    if mask[-1]:
        en.append(len(mask))
    return list(zip(st, en))


for grp, dur, files in PAIRS:
    print("\n" + "=" * 76)
    print("### %s组 · 跨度 %.0fs · 停顿门限 −40dB、≥0.25s ###" % (grp, dur))
    for nm, p in files:
        y = load(p, "L%s%s" % (grp, nm.replace(" ", "")), dur)
        if y is None:
            print("  %-12s 缺文件 %s" % (nm, p))
            continue
        db = env_db(y)
        d = len(db) * HOP
        ps = sorted([(e - s) * HOP for s, e in runs(db < -40.0)
                     if (e - s) * HOP >= 0.25], reverse=True)
        print("\n  == %-12s %.0fs ==  ≥0.25s 停顿 %d 处（%.2f 处/分）"
              % (nm, d, len(ps), len(ps) / d * 60))
        b = {"0.25~0.4": 0, "0.4~0.6": 0, "0.6~1.0": 0, "≥1.0": 0}
        for v in ps:
            if v < 0.4:
                b["0.25~0.4"] += 1
            elif v < 0.6:
                b["0.4~0.6"] += 1
            elif v < 1.0:
                b["0.6~1.0"] += 1
            else:
                b["≥1.0"] += 1
        print("     " + "  ".join("%s: %d 处" % (k, v) for k, v in b.items()))
        print("     最长的 8 处: %s" % "  ".join("%.2fs" % v for v in ps[:8]))
