# -*- coding: utf-8 -*-
"""headfix 收官验证：
  1) 202610190902（L3 验收文件）切后 ASR 无损对比
  2) 切前 vs 切后音色（campplus，cosy venv）
  3) EL 标杆头部结构 vs 我们缺陷文件头部结构
"""
import importlib.util
import subprocess

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

spec = importlib.util.spec_from_file_location("hx", "/mnt/tts/diag/q5_headfix.py")
hx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hx)

m = WhisperModel("Systran/faster-whisper-small", device="cpu",
                 compute_type="int8", local_files_only=True)

# 1) L3 文件切后 ASR（前 12s 窗口对比）
p = "/mnt/tts/diag/v0902/202610190902-1.mp3.16k.wav"
x, sr = sf.read(p, dtype="float32")
if x.ndim > 1:
    x = x.mean(axis=1)
y, d = hx.fix_head(x, sr)
print("=== L3 文件 %s ===" % p.split("/")[-1])
print("诊断: cut=%.2f S=%s groups=%s dur=%.3f" % (d["cut"], d["S"], d["groups"], d["frag_dur"]))
sf.write("/mnt/tts/diag/v0902/l3fix_cut.16k.wav", y, sr)
for tag, seg in (("原[0-12s]", x[:12 * sr]), ("切后[0-12s]", y[:12 * sr])):
    txt = " ".join(s.text for s in m.transcribe(seg, language="ko",
                                                vad_filter=False)[0]).strip()
    print("  %s: %s" % (tag, txt[:100]))

# 2) 音色对比（campplus 16k，同人阈值 0.75）
print("\n=== 音色（切前 vs 切后）===")
sf.write("/mnt/tts/diag/v0902/l3fix_orig.16k.wav", x, sr)
r = subprocess.run(
    ["/mnt/tts/projects/cosy/venv/bin/python", "/mnt/tts/diag/spkcmp53.py",
     "/mnt/tts/diag/v0902/l3fix_orig.16k.wav",
     "/mnt/tts/diag/v0902/l3fix_cut.16k.wav"],
    capture_output=True, text=True, timeout=600)
print(r.stdout.strip()[-400:] or r.stderr.strip()[-400:])

# 3) EL 标杆头部结构
print("\n=== 头部结构对比（0-1.2s 包络）===")
for tag, p in (("EL标杆", "/mnt/tts/diag/v0902/EL_female.wav.16k.wav"),
               ("我们-L3", p)):
    xx, ss = sf.read(p, dtype="float32")
    if xx.ndim > 1:
        xx = xx.mean(axis=1)
    v = hx.envdb(xx, ss)
    show = " ".join("%.2f:%+.0f" % (i * 0.01, v[i]) for i in range(0, min(len(v), 120), 4))
    print("  %-8s %s" % (tag, show))
