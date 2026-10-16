#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""① 定位残余 <−100dB 帧；② 对比 EL 与我们修后的「−52dB 平躺平台」长度分布。"""
import os
import subprocess

import numpy as np
import soundfile as sf

SR = 24000
HOP = 0.005
TMP = "/mnt/tts/diag/_silfix2"
os.makedirs(TMP, exist_ok=True)
FILES = [
    ("EL女声原声", "/mnt/tts/data/cases/52727ec43154/merged.wav"),
    ("我们现役", "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"),
    ("地板-52", os.path.join(TMP, "f_52.mp3")),
]


def load(src, tag):
    o = os.path.join(TMP, "_p%s.wav" % tag)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ac", "1",
                    "-ar", str(SR), o], check=True)
    return sf.read(o, dtype="float32")[0]


def env_db(y):
    h = int(HOP * SR)
    n = len(y) // h
    e = np.sqrt(np.array([(y[i * h:(i + 1) * h] ** 2).mean() for i in range(n)]))
    return 20 * np.log10(e + 1e-12)


def runs(m):
    d = np.diff(m.astype(np.int8))
    st = list(np.where(d == 1)[0] + 1)
    en = list(np.where(d == -1)[0] + 1)
    if m[0]:
        st.insert(0, 0)
    if m[-1]:
        en.append(len(m))
    return list(zip(st, en))


for name, p in FILES:
    y = load(p, name.replace(" ", ""))
    db = env_db(y)
    print("\n== %s (%.1fs) ==" % (name, len(y) / SR))
    # 平躺平台：连续帧落在 -56~-46dB
    st = [(e - s) * HOP for s, e in runs((db > -56) & (db < -46))]
    st = sorted([v for v in st if v >= 0.02], reverse=True)
    if st:
        print("  平台(−56~−46dB) %d 段：最长 %.2fs  前5 %s  合计 %.1fs"
              % (len(st), st[0], ["%.2f" % v for v in st[:5]], sum(st)))
    # 更宽：-60~-45dB
    st2 = [(e - s) * HOP for s, e in runs((db > -60) & (db < -45))]
    st2 = sorted([v for v in st2 if v >= 0.02], reverse=True)
    if st2:
        print("  平台(−60~−45dB) %d 段：最长 %.2fs  前5 %s" % (len(st2), st2[0],
                                                            ["%.2f" % v for v in st2[:5]]))
    # 残余数字零
    z = np.where(db < -100)[0]
    if len(z):
        gs = runs(db < -100)
        print("  🔴 <−100dB 帧 %d 个，成段：%s"
              % (len(z), ["%.2f~%.2fs(%.0fms)" % (s * HOP, e * HOP, (e - s) * HOP * 1000)
                          for s, e in gs]))
    else:
        print("  ✅ 无 <−100dB 帧")
