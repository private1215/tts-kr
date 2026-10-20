# -*- coding: utf-8 -*-
"""① 碎片段内容 ASR ② 变量分离对照实验：
   A: EL_female + cases 文案(반갑습니다 아모텍) x3  -> 文案依赖?
   C: 男声 + 로보티즈 文案 x3                      -> 音色依赖?
   D: EL_female + 纯 "로보티즈." x3                 -> 最小触发?
"""
import importlib.util
import sys

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

spec = importlib.util.spec_from_file_location("hx", "/mnt/tts/diag/q5_headfix.py")
hx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hx)

m = WhisperModel("Systran/faster-whisper-small", device="cpu",
                 compute_type="int8", local_files_only=True)

# ① 碎片内容
print("=== 碎片段 ASR ===")
for p, a, b in (("/mnt/tts/diag/v0902/202610190902-1.mp3.16k.wav", 0.10, 0.75),
                ("/mnt/tts/diag/v0902/headrepro-1019-6.16k.wav", 0.10, 0.45),
                ("/mnt/tts/diag/v0902/headrepro-1019-8.16k.wav", 0.10, 0.72)):
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    seg = x[int(a * sr):int(b * sr)]
    txt = " ".join(s.text for s in m.transcribe(seg, language="ko",
                                                vad_filter=False)[0]).strip()
    print("  %s [%.2f-%.2f] -> %r" % (p.split("/")[-1], a, b, txt))

# ② 变量分离实验
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
import os
os.environ.setdefault("TTS_BASE", "http://127.0.0.1:8000")
from tools.eval5 import submit_job, wait_job, login  # noqa: E402

login()
CASE_TXT = ("반갑습니다, 아모텍 주주 여러분. 이민정입니다. 사실 우리 아모텍 주주님들, "
            "그동안 속앓이 진짜 심하셨잖아요, 고점에서 크게 두들려 맞고 바닥을 길게 기면서, "
            "도대체 언제 턴어라운드하나 답답하셨을 겁니다.")
ROBO_TXT = ("로보티즈 주주 여러분, 반갑습니다. 정민철 프로입니다. "
            "오늘 우리 주주님들께, 진짜 피가 되고 살이 되는 중요한 말씀 드리려고, "
            "급하게 카메라 켰습니다.")

EXPS = [("A-case-f", "EL_female", CASE_TXT, 3),
        ("C-robo-m", "EL_male", ROBO_TXT, 3),
        ("D-robo-only", "EL_female", "로보티즈.", 3)]

import glob
import subprocess
for name, voice, text, n in EXPS:
    for i in range(n):
        title = "headvar-%s-%d" % (name, i)
        r = submit_job(text, voice, title)
        print("submit %s -> %s" % (title, r.get("id", r)))
