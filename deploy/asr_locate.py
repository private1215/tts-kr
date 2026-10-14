#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""asr_locate.py — 用 faster-whisper 定位号码段（逐字时间戳），对比 EL 与我们"""
import subprocess, tempfile, os, sys, json
import numpy as np, soundfile as sf

MODEL_ID = "Systran/faster-whisper-small"

def cut16(path, ss, to):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg","-y","-v","error","-ss",str(ss),"-to",str(to),
                    "-i",path,"-ar","16000","-ac","1",t], check=True)
    y, sr = sf.read(t, dtype="float32"); os.unlink(t)
    return y

def main():
    from faster_whisper import WhisperModel
    m = WhisperModel(MODEL_ID, device="cpu", compute_type="int8",
                     cpu_threads=4, local_files_only=True)
    jobs = [
        ("OURS_X2f_v3f", "/mnt/tts/diag/cross-20261013/X2f_v3f.wav", 190.0, 238.0),
        ("EL_5272",      "/mnt/tts/data/cases/52727ec43154/merged.wav", 195.0, 243.0),
    ]
    DIGITS = set("공일육칠이사삼오육칠팔구영하나둘셋넷다섯여섯")
    for name, path, ss, to in jobs:
        y = cut16(path, ss, to)
        segs, info = m.transcribe(y, language="ko", beam_size=1,
                                  vad_filter=False, word_timestamps=True)
        print(f"\n=== {name} [{ss}~{to}s] ===")
        words = []
        for s in segs:
            print(f"  seg[{s.start:6.2f}-{s.end:6.2f}] {s.text.strip()}")
            if s.words:
                for w in s.words:
                    words.append((round(s.start + w.start,2), round(s.start + w.end,2), w.word.strip()))
        print("  -- 词级时间戳 --")
        for st, en, w in words:
            print(f"    {st:6.2f}-{en:6.2f} {w}")
        # 找疑似数字逐位读的连续串
        print("  -- 疑似数字位（含공/일/육/칠/이/사/삼等）--")
        cand = [(st,en,w) for st,en,w in words if any(ch in DIGITS for ch in w) and len(w) <= 4]
        for st,en,w in cand:
            print(f"    {st:6.2f}-{en:6.2f} {w}")

main()
