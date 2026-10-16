#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""两个决定性检查（同文案同跨度 198.6s）
A. 停顿清册：按时长分桶数数量 + 总时长，我们 vs EL。
   —— prate 已显示我们 24.5~26.2 处/分 vs EL 22.05，员工原话就是「断句有异声」。
B. 员工点名的四处，到底有没有**真的多念出一个「额/啊」**？
   用 whisper-medium 逐词时间戳，我们 vs EL 同位置并排。这是"多念字"的唯一直接证据。
"""
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

SR = 24000
HOP = 0.010
SPAN = 198.6
TMP = "/mnt/tts/diag/_z46"
os.makedirs(TMP, exist_ok=True)
A = "/mnt/tts/diag/xf5272/prod_before_silfloor.mp3"
B = "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"
EL = "/mnt/tts/data/cases/52727ec43154/merged.wav"
POS = [96.0, 131.0, 145.0, 170.0]
BUCKETS = [(0.10, 0.20), (0.20, 0.30), (0.30, 0.50), (0.50, 1.00), (1.00, 99.0)]


def load(src, tag, dur=None):
    o = os.path.join(TMP, "%s.wav" % tag)
    c = ["ffmpeg", "-y", "-v", "error"]
    if dur:
        c += ["-t", "%.2f" % dur]
    c += ["-i", src, "-ac", "1", "-ar", str(SR), o]
    subprocess.run(c, check=True)
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


def pauses(db, thr=-45.0):
    return [((e - s) * HOP, s * HOP) for s, e in runs(db < thr)]


print("### A. 停顿清册（同文案前 198.6s，门限 −45dB）###")
for nm, p in (("我们·修前", A), ("我们·修后", B), ("EL 原声", EL)):
    db = env_db(load(p, "A%s" % nm.replace(" ", ""), SPAN))
    du = len(db) * HOP
    ps = pauses(db)
    tot = sum(x[0] for x in ps)
    print("\n  == %-10s %.1fs ==  全部 %d 处  %.1f 处/分  合计 %.1fs（%.2f%%）"
          % (nm, du, len(ps), len(ps) / du * 60, tot, 100 * tot / du))
    for lo, hi in BUCKETS:
        b = [x for x in ps if lo <= x[0] < hi]
        print("     %4.2f~%4.2fs : %3d 处  合计 %5.2fs  平均 %5.3fs"
              % (lo, hi, len(b), sum(x[0] for x in b),
                 (sum(x[0] for x in b) / len(b)) if b else 0))
    b = [x for x in ps if x[0] >= 0.28]
    print("     ≥0.28s（prate 口径）: %3d 处  %.1f 处/分" % (len(b), len(b) / du * 60))

print("\n\n### B. 员工点名四处：我们 vs EL 逐词对照（whisper-medium）###")
from faster_whisper import WhisperModel
M = WhisperModel("medium", device="cpu", compute_type="int8", local_files_only=True)


def words(src, t0, dur=3.0, tag="x"):
    o = os.path.join(TMP, "%s.wav" % tag)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "%.2f" % (t0 - dur / 2),
                    "-t", "%.2f" % dur, "-i", src, "-ac", "1", "-ar", "16000", o],
                   check=True)
    y = sf.read(o, dtype="float32")[0]
    y = (y / (np.abs(y).max() + 1e-9)).astype(np.float32)
    s, _ = M.transcribe(y, language="ko", beam_size=5, vad_filter=False,
                        word_timestamps=True, condition_on_previous_text=False)
    out = []
    for seg in s:
        for w in (seg.words or []):
            out.append((t0 - dur / 2 + w.start, w.word.strip()))
    return out


for t in POS:
    print("\n== 绝对 %.1fs ==（两者都是同一句话的同一处）" % t)
    for nm, p in (("我们·修前", A), ("EL 原声", EL)):
        ws = words(p, t, 3.0, "%s%d" % (nm[:2], int(t)))
        print("  %-8s %s" % (nm, "  ".join("%.2f:%s" % (tt - t, w) for tt, w in ws)))
print("\nDONE")
