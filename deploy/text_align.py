#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""我们这个 198.6s 成品，与 EL 女声 525s 原声，开头是不是**同一句话**？

如果开头不同 → 两者是不同文案，"同一绝对秒 = 同一句话"的对照口径全部作废，
所有拿 EL 当基线的逐秒对比（含 blip_el_cmp）都得作废。
"""
import os
import subprocess

import numpy as np
import soundfile as sf

SR = 16000
TMP = "/mnt/tts/diag/_talign"
os.makedirs(TMP, exist_ok=True)
A = "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"
B = "/mnt/tts/data/cases/52727ec43154/merged.wav"
DUR = 15.0


def seg(src, tag, t0=0.0, dur=DUR):
    o = os.path.join(TMP, "%s.wav" % tag)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ss", "%.2f" % t0,
                    "-t", "%.2f" % dur, "-ac", "1", "-ar", str(SR), o], check=True)
    y = sf.read(o, dtype="float32")[0]
    return ((y - y.mean()) / (np.abs(y).max() + 1e-9)).astype(np.float32)


def asr(y, tag):
    from faster_whisper import WhisperModel
    m = WhisperModel("medium", device="cpu", compute_type="int8",
                     local_files_only=True)
    s, _ = m.transcribe(y, language="ko", beam_size=5, vad_filter=False,
                        condition_on_previous_text=False)
    return " ".join(x.text.strip() for x in s)


for nm, p in (("我们成品", A), ("EL原声", B)):
    print("== %s 前 %.0fs ==" % (nm, DUR))
    print(asr(seg(p, nm.replace(" ", "")), nm))
    print()
