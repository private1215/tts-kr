# -*- coding: utf-8 -*-
"""grunt53b.py —— 定位「极短嗯」是模型产生还是后处理引入（16k 统一口径）。"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 0.005


def load(p):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(SR), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    return x


def grunts(x):
    n = int(FRAME * SR)
    nfr = len(x) // n
    rms = np.sqrt((x[:nfr * n].reshape(nfr, n) ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-12)
    voiced = db > np.percentile(db, 35)
    segs, i = [], 0
    while i < len(voiced):
        if voiced[i]:
            j = i
            while j < len(voiced) and voiced[j]:
                j += 1
            segs.append((i * FRAME, (j - i) * FRAME))
            i = j
        else:
            i += 1
    out = []
    for k in range(1, len(segs) - 1):
        s, d = segs[k]
        if 0.02 <= d <= 0.12:
            pe = segs[k - 1][0] + segs[k - 1][1]
            nx = segs[k + 1][0]
            if (s - pe) > 0.15 and (nx - (s + d)) > 0.15:
                out.append((s, d))
    return out


FILES = [
    ("裸输出 chunk15", "/mnt/tts/diag/omnichunk53/chunk15.wav"),
    ("裸输出 EL参考", "/mnt/tts/diag/refswapfull53_male.wav"),
    ("EL参考+fixchain", "/mnt/tts/diag/refchain53/m_elref.wav"),
    ("现役成品(mp3)", "/mnt/tts/output/l3-m-omni/l3-m-omni-1.mp3"),
    ("男 EL", "/mnt/tts/data/cases/d172b8c4efb3/merged.wav"),
]
print("%-20s %8s %10s %12s" % ("文件", "极短嗯", "次/分钟", "位置s"))
for name, p in FILES:
    if not os.path.exists(p):
        print("%-20s 缺失" % name)
        continue
    x = load(p)
    g = grunts(x)
    dur = len(x) / SR
    print("%-20s %8d %10.2f   %s" % (name, len(g), len(g) / dur * 60,
                                     " ".join("%.1f" % s for s, _ in g[:12])))
