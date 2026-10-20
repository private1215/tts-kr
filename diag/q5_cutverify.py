# -*- coding: utf-8 -*-
"""CUT 文件 ASR 无损验证 + #7 形态查看"""
import numpy as np, soundfile as sf
from faster_whisper import WhisperModel

m = WhisperModel("Systran/faster-whisper-small", device="cpu",
                 compute_type="int8", local_files_only=True)
SR = 16000

import importlib.util
spec = importlib.util.spec_from_file_location("hx", "/mnt/tts/diag/q5_headfix.py")
hx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hx)


def asr(x):
    segs, _ = m.transcribe(x, language="ko", vad_filter=False)
    return " ".join(s.text for s in segs).strip()


for k in (6, 7, 8, 9):
    p = "/mnt/tts/diag/v0902/headrepro-1019-%d.16k.wav" % k
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    v = hx.envdb(x)
    y, d = hx.fix_head(x)
    print("=== #%d ===" % k)
    # 头部包络 0-2.2s
    show = " ".join("%.2f:%+.0f" % (i * 0.01, v[i])
                    for i in range(0, min(len(v), 220), 2))
    print("  ENV %s" % show)
    print("  诊断: %s" % d)
    if d.get("reason") == "cut":
        t0 = asr(x)
        t1 = asr(y)
        print("  原ASR: %s" % t0[:60])
        print("  切后ASR: %s" % t1[:60])
        print("  一致: %s" % (t0.replace(" ", "")[:40] == t1.replace(" ", "")[:40]))
    else:
        print("  原ASR: %s" % asr(x)[:60])
