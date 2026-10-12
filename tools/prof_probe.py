# -*- coding: utf-8 -*-
"""画像提取耗时探针：定位到底哪一步慢（临时诊断脚本）。"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

import case_style_profile as CSP

t0 = time.time()
cases = CSP.list_cases()
print("list_cases %.1fs  n=%d" % (time.time() - t0, len(cases)), flush=True)

cid, meta = cases[0]
wav = os.path.join(CSP.CASES_DIR, cid, "merged.wav")
print("probe case:", cid, wav, os.path.getsize(wav) / 1e6, "MB", flush=True)

t = time.time()
txt = CSP.case_text(cid, meta)
print("case_text %.1fs chars=%s" % (time.time() - t, len(txt or "")), flush=True)

import soundfile as sf
t = time.time()
x, sr = CSP.PC.PR.load(wav)
print("PR.load %.1fs  %.1fs@%d" % (time.time() - t, len(x) / sr, sr), flush=True)

t = time.time()
segs, cuts, hop = CSP.PC.PR.sent_windows(x, sr)
print("sent_windows %.1fs segs=%d cuts=%d" % (time.time() - t, len(segs), len(cuts)),
      flush=True)

t = time.time()
tf = CSP.proso.tier_of(x, sr, is_male=False)
print("praat tier_of %.1fs" % (time.time() - t), flush=True)

t = time.time()
p = CSP.profile_of(wav, False, txt)
print("profile_of %.1fs -> %s" % (time.time() - t, p), flush=True)
print("TOTAL %.1fs" % (time.time() - t0), flush=True)
