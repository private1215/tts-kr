# -*- coding: utf-8 -*-
"""把 GS 男声的 ASR 假设文本与原文逐句对照，看 CER 0.19 到底是真念错
还是 whisper 把韩文数字写成阿拉伯数字导致的假错。
"""
import sys, json, difflib
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
from app import asr_check

jobs = json.load(open("/mnt/tts/data/jobs.json"))
REF = jobs["7d8af90f3ffa"]["text"]

for name in ("E_cur_10s_160", "F_p1_5.45s_40", "G_p2_7.55s_51"):
    p = f"/mnt/tts/diag/gsab47b/{name}.wav"
    r = asr_check.check_chunk(p, REF, "Korean")
    hyp = r.get("hyp", "")
    print(f"\n{'='*70}\n{name}  CER={r.get('cer')}")
    print(f"-- 原文({len(REF)}字): {REF}")
    print(f"-- ASR ({len(hyp)}字): {hyp}")
    a, b = asr_check.normalize(REF), asr_check.normalize(hyp)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    print("-- 差异块:")
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        print(f"   {tag:<8} 原文[{i1}:{i2}]={a[i1:i2]!r}  ASR[{j1}:{j2}]={b[j1:j2]!r}")
