# -*- coding: utf-8 -*-
"""长文件头部修复验证：cases/merged.wav（24k, 7.5-9min）
对每个文件：
  1) find_cut 判据
  2) CUT 则对比 原头部 ASR vs 切后头部 ASR（3s 窗口）
     - 原 ASR 多出幻听词（자/그러니까 类）= 碎片是垃圾，切对了
     - 两者一致 = 无损益
  3) click 测量
"""
import glob
import importlib.util

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

spec = importlib.util.spec_from_file_location("hx", "/mnt/tts/diag/q5_headfix.py")
hx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hx)

m = WhisperModel("Systran/faster-whisper-small", device="cpu",
                 compute_type="int8", local_files_only=True)

files = sorted(glob.glob("/mnt/tts/data/cases/*/merged.wav"))
# 取 12 条（优先长文件）
files = files[:12]

n_cut = 0
for p in files:
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    y, d = hx.fix_head(x, sr)
    tag = p.split("/")[-2][:8]
    dur = len(x) / sr
    if d.get("reason") != "cut":
        print("%s %6.1fs keep reason=%-14s S=%s groups=%s"
              % (tag, dur, d.get("reason"), d.get("S"), d.get("groups")))
        continue
    n_cut += 1
    # 头部 ASR 对比：原 [0, cut+2.5s] vs 切后 [0, 2.5s]
    seg_o = x[:int((d["cut"] + 2.5) * sr)]
    seg_c = y[:int(2.5 * sr)]
    to = " ".join(s.text for s in m.transcribe(seg_o, language="ko",
                                               vad_filter=False)[0]).strip()
    tc = " ".join(s.text for s in m.transcribe(seg_c, language="ko",
                                               vad_filter=False)[0]).strip()
    c0 = hx.click_count(x[:int(5 * sr)])
    c1 = hx.click_count(y[:int(5 * sr)])
    print("%s %6.1fs CUT  cut=%.2f S=%s groups=%s dur=%.3f click %d->%d"
          % (tag, dur, d["cut"], d["S"], d["groups"], d.get("frag_dur", -1), c0, c1))
    print("    原头部ASR: %s" % to[:70])
    print("    切后 ASR: %s" % tc[:70])

print("\n总计 %d 条，CUT %d 条 (%.0f%%)" % (len(files), n_cut, 100.0 * n_cut / len(files)))
