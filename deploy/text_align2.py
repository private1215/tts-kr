#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐点对齐校验：我们 198.6s 成品的第 95s / 131s / 170s，与 EL 525s 原声同秒是否同一句？

开头 15s 已证逐字一致。本脚本验证中后段是否仍对齐（若我们对齐但更快，之后就会漂）。
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
T0S = [95.0, 131.0, 170.0, 183.0]
DUR = 12.0
M = None


def seg(src, tag, t0):
    o = os.path.join(TMP, "%s.wav" % tag)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ss", "%.2f" % t0,
                    "-t", "%.2f" % DUR, "-ac", "1", "-ar", str(SR), o], check=True)
    y = sf.read(o, dtype="float32")[0]
    return ((y - y.mean()) / (np.abs(y).max() + 1e-9)).astype(np.float32)


def asr(y):
    global M
    if M is None:
        from faster_whisper import WhisperModel
        M = WhisperModel("medium", device="cpu", compute_type="int8",
                         local_files_only=True)
    s, _ = M.transcribe(y, language="ko", beam_size=5, vad_filter=False,
                        condition_on_previous_text=False)
    return " ".join(x.text.strip() for x in s)


for t0 in T0S:
    print("===== 第 %.0fs 起 %.0fs（±0.3s） =====" % (t0, DUR))
    for nm, p in (("我们", A), ("EL  ", B)):
        # 我们成品只有 198.6s，超出就跳过
        if nm == "我们" and t0 + DUR > 198.0:
            print("%s: 超出文件长度，跳过" % nm)
            continue
        print("%s: %s" % (nm, asr(seg(p, "%s_%d" % (nm.strip(), t0), t0))))
    print()
