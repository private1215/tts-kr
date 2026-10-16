#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端回归：修前成品 vs 修后成品(链路已带 silfloor) vs EL 原声，三方停顿口径对照。"""
import os
import subprocess

import numpy as np
import soundfile as sf

SR = 24000
HOP = 0.005
TMP = "/mnt/tts/diag/_cmp44"
os.makedirs(TMP, exist_ok=True)
FILES = [
    ("EL女声原声(525s)", "/mnt/tts/data/cases/52727ec43154/merged.wav"),
    ("修前成品", "/mnt/tts/diag/xf5272/prod_before_silfloor.mp3"),
    ("修后成品", "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"),
]


def load(src, tag):
    o = os.path.join(TMP, "%s.wav" % tag)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ac", "1",
                    "-ar", str(SR), o], check=True)
    return sf.read(o, dtype="float32")[0]


def env_db(y):
    h = int(HOP * SR)
    n = len(y) // h
    m = y[:n * h].astype(np.float64).reshape(n, h)
    return 20 * np.log10(np.sqrt((m ** 2).mean(axis=1)) + 1e-12)


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
    lo = db[db < -45]
    q = np.percentile(lo, [1, 5, 10, 50]) if lo.size else [0] * 4
    vs = [(s, e) for s, e in runs(db < -60) if (e - s) * HOP >= 0.04]
    print("== %-16s %.1fs ==" % (name, len(y) / SR))
    print("   p1 %6.1f | p5 %6.1f | p10 %6.1f | 中位 %6.1f | min %7.1f | <−100dB %4d 帧 "
          "| 深谷 %3d 段 | RMS %6.1fdB"
          % (q[0], q[1], q[2], q[3], db.min(), int((db < -100).sum()), len(vs),
             20 * np.log10(np.sqrt((y ** 2).mean()) + 1e-12)))
    # 员工点名三处（绝对秒同句已证）
    print("   点名处最低：%s" % "  ".join(
        "%.1fs=%6.0f" % (m, db[int((m - 0.7) / HOP):int((m + 0.7) / HOP)].min())
        for m in (95.5, 131.9, 170.5)))
