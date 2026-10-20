# -*- coding: utf-8 -*-
"""L1 实测：生产 API 提交 로보티즈 文案，验证 headfix53 上线效果"""
import glob
import os
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
os.environ.setdefault("TTS_BASE", "http://127.0.0.1:8000")
from tools.eval5 import submit_job, login  # noqa: E402

import importlib.util
spec = importlib.util.spec_from_file_location("hx", "/mnt/tts/diag/q5_headfix.py")
hx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hx)

TEXT = ("로보티즈 주주 여러분, 반갑습니다. 정민철 프로입니다. "
        "오늘 우리 주주님들께, 진짜 피가 되고 살이 되는 중요한 말씀 드리려고, "
        "급하게 카메라 켰습니다. 와, 지금 로보티즈, 자리 기가 막히거든요. "
        "왜 이 자리가 중요하냐고요? 시월 이일, 오전 열한 시 삼십사분 기준, "
        "로보티즈 삼십만구천오백원. 전일 대비 한 사쩜사퍼센트 정도 띄워주고 있습니다.")

login()
ids = []
for i in range(3):
    r = submit_job(TEXT, "EL_female", "l1headfix-%d" % i)
    ids.append(r["id"])
    print("submit l1headfix-%d -> %s" % (i, r["id"]))

# 等完成
deadline = time.time() + 900
done = {}
while time.time() < deadline and len(done) < len(ids):
    done = {}
    for t in ("l1headfix-0", "l1headfix-1", "l1headfix-2"):
        cands = sorted(glob.glob("/mnt/tts/output/%s/*.mp3" % t),
                       key=os.path.getmtime, reverse=True)
        if cands:
            done[t] = cands[0]
    if len(done) == 3:
        break
    print("等待 %d/3 ..." % len(done), flush=True)
    time.sleep(15)

print("\n=== 成品头部检查 ===")
for t in sorted(done):
    mp3 = done[t]
    w = "/mnt/tts/diag/v0902/%s.16k.wav" % t
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp3,
                    "-ar", "16000", "-ac", "1", w], check=True)
    x, sr = sf.read(w, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    v = hx.envdb(x, sr)
    S = hx.speech_onset(v)
    # 头部包络
    show = " ".join("%.2f:%+.0f" % (i * 0.01, v[i])
                    for i in range(0, min(len(v), 100), 4))
    print("%s  S=%.2fs  时长 %.1fs" % (t, S or -1, len(x) / sr))
    print("   ENV %s" % show)
