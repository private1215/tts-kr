# -*- coding: utf-8 -*-
"""q1_template_24k_tail.py —— Q1 补充：模版句末粘连嗯在原生 24kHz 下复测。

check_template.py 是在 16k（重采样后）测的 2 个。这里用原生 24k 复测，
确认「嗯」不是重采样伪影；同时测我们成品与 EL 标杆作对照。
"""
import os
import subprocess

import numpy as np
import soundfile as sf

SR = 24000
HOP = int(0.005 * SR)
FILES = [
    ("模版 EL_female.wav（原生24k）", "/mnt/tts/data/voices/EL_female.wav", True),
    ("我们成品 202610190902（16k）", "/mnt/tts/diag/v0902/202610190902-1.mp3.16k.wav", False),
    ("EL 标杆 merged（16k）", "/mnt/tts/diag/v0902/hm_merged.wav.wav", False),
]


def load(path, native):
    if native:
        x, sr = sf.read(path, dtype="float32")
    else:
        w = "/tmp/q1t_%s.wav" % os.path.basename(path)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                        "-ar", "16000", "-ac", "1", w], check=True)
        x, sr = sf.read(w, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def tails(x, sr):
    hop = int(0.005 * sr)
    n = len(x) // hop
    rms = np.array([np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2) + 1e-12)
                    for i in range(n)])
    db = 20 * np.log10(rms + 1e-12)
    voiced = db > -40
    segs = []
    i = 0
    while i < n:
        if voiced[i]:
            j = i
            while j < n and voiced[j]:
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1
    out = []
    for k in range(1, len(segs)):
        pi, pj = segs[k - 1]
        ci, cj = segs[k]
        if (ci - pj) * 0.005 >= 0.03:
            continue
        if not (0.005 <= (cj - ci) * 0.005 <= 0.12):
            continue
        if (pj - pi) * 0.005 <= 0.10:
            continue
        # 🔴 tailgrunt53e 完整判据还要求：该段之后必须还有 >0.05s 的静音间隙
        # （即它确实是「句末」，而不是句中换气）
        if k + 1 >= len(segs):
            continue
        if (segs[k + 1][0] - cj) * 0.005 <= 0.05:
            continue
        s = x[ci * hop:cj * hop]
        if len(s) < 8:
            continue
        pk = float(np.max(np.abs(s)))
        pkdb = 20 * np.log10(pk + 1e-12)
        if not (-45 <= pkdb <= -20):
            continue
        zcr = float(np.mean(np.abs(np.diff(np.sign(s))) > 0))
        S = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
        fr = np.fft.rfftfreq(len(s), 1.0 / sr)
        lowf = float(S[fr < 300].sum() / (S.sum() + 1e-12))
        if zcr < 0.15 and lowf > 0.45:
            out.append((ci * 0.005, (cj - ci) * 0.005, pkdb, zcr, lowf))
    return out


for lab, p, nat in FILES:
    x, sr = load(p, nat)
    dur = len(x) / sr
    t = tails(x, sr)
    print("%s：时长 %.2f s，句末粘连嗯 %d 个（%.2f/分）" % (lab, dur, len(t), len(t) / (dur / 60)))
    for t0, ln, pkdb, zcr, lowf in t:
        print("    t=%7.2f  %3.0fms  峰值%.0fdB  zcr=%.3f  低频=%.2f"
              % (t0, ln * 1000, pkdb, zcr, lowf))
    print()
