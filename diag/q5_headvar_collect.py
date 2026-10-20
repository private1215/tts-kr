# -*- coding: utf-8 -*-
"""收集 headvar 实验结果并测头部缺陷"""
import glob
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
os.environ.setdefault("TTS_BASE", "http://127.0.0.1:8000")
from tools.eval5 import wait_job  # noqa: E402

import importlib.util
spec = importlib.util.spec_from_file_location("hx", "/mnt/tts/diag/q5_headfix.py")
hx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hx)

TITLES = []
for name, n in (("A-case-f", 3), ("C-robo-m", 3), ("D-robo-only", 3)):
    for i in range(n):
        TITLES.append("headvar-%s-%d" % (name, i))

# 等所有 job 完成
import json
jobs = {}
for t in TITLES:
    # 从 job 列表找（eval5 的 wait_job 需要 job id；这里直接扫输出目录）
    pass

# 直接扫输出目录（按 title 前缀，取 mtime 最新的目录）
OUT = "/mnt/tts/data/output"
deadline = time.time() + 900
found = {}
while time.time() < deadline:
    found = {}
    for t in TITLES:
        cands = sorted(glob.glob(os.path.join(OUT, t + "*")),
                       key=os.path.getmtime, reverse=True)
        for c in cands:
            wavs = glob.glob(os.path.join(c, "*.wav")) + \
                   glob.glob(os.path.join(c, "*.mp3"))
            if wavs:
                found[t] = c
                break
    if len(found) == len(TITLES):
        break
    print("等待中 %d/%d ..." % (len(found), len(TITLES)), flush=True)
    time.sleep(20)

print("\n=== 结果 ===")
for t in TITLES:
    if t not in found:
        print("%s 未找到输出" % t)
        continue
    c = found[t]
    # 优先 16k wav（后处理中间产物），否则 mp3
    w = sorted(glob.glob(os.path.join(c, "*.16k.wav")),
               key=os.path.getmtime, reverse=True)
    if not w:
        w = sorted(glob.glob(os.path.join(c, "*.wav")),
                   key=os.path.getmtime, reverse=True)
    if not w:
        w = sorted(glob.glob(os.path.join(c, "*.mp3")),
                   key=os.path.getmtime, reverse=True)
    if not w:
        print("%s 目录无音频: %s" % (t, os.listdir(c)))
        continue
    p = w[0]
    try:
        x, sr = sf.read(p, dtype="float32")
    except Exception as e:
        print("%s 读取失败 %s: %s" % (t, p, e))
        continue
    if x.ndim > 1:
        x = x.mean(axis=1)
    y, d = hx.fix_head(x, sr)
    if d.get("reason") == "cut":
        print("%-22s CUT  cut=%.2f S=%s groups=%s dur=%.3f  (%.1fs)"
              % (t, d["cut"], d["S"], d["groups"], d.get("frag_dur", -1),
                 len(x) / sr))
    else:
        print("%-22s keep reason=%-14s S=%s (%.1fs)"
              % (t, d.get("reason"), d.get("S"), len(x) / sr))
