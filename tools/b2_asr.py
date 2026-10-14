# -*- coding: utf-8 -*-
"""第二批：交付前格式归一（统一 24k/单声道/PCM16）+ ASR 内容筛查。

· 归一只为**盲听时播放条件一致**（首轮交付件全是 24k 单声道），
  不做响度/EQ/停顿等任何处理；GS 原始产物是 32k，这里只做采样率与声道归一。
· ASR 仍然只是「有没有少读、有没有跑飞」的内容线索，不作听感或音色结论。

用法（CPU）：
    python -u tools/b2_asr.py
"""
import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
from app.asr_check import cer, transcribe  # noqa: E402

STATE = "/mnt/tts/diag/b2-20261013"
DELIVER = os.path.join(STATE, "deliver")
CASES = "/mnt/tts/data/cases"

# 服务器产物 -> 盲标（盲标本身不暴露引擎）
ITEMS = [
    ("case1-cand6", "C1.wav", "ce1a3fda551a"),
    ("case1-cand7", "G1.wav", "ce1a3fda551a"),
    ("case2-cand6", "C2.wav", "d172b8c4efb3"),
    ("case2-cand7", "G2.wav", "d172b8c4efb3"),
]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    os.makedirs(DELIVER, exist_ok=True)
    out = {}
    for blind, src, cid in ITEMS:
        srcp = os.path.join(STATE, src)
        dst = os.path.join(DELIVER, blind + ".wav")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", srcp,
                        "-ac", "1", "-ar", "24000", "-sample_fmt", "s16", dst],
                       check=True)
        dur = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", dst], capture_output=True, text=True
        ).stdout.strip())
        ref = open(os.path.join(CASES, cid, "full.txt"), encoding="utf-8").read()
        t0 = time.time()
        hyp = transcribe(dst, "ko")
        out[blind] = {
            "盲标": blind,
            "案例": cid,
            "源产物": srcp,
            "源产物sha256": sha(srcp),
            "交付文件": dst,
            "交付sha256": sha(dst),
            "交付时长_秒": round(dur, 2),
            "归一处理": "ffmpeg -ac 1 -ar 24000 -sample_fmt s16（仅采样率/声道归一，未做其它处理）",
            "ref_chars": len(ref),
            "hyp_chars": len(hyp),
            "hyp_head": hyp[:120],
            "cer": cer(ref, hyp),
            "asr_seconds": round(time.time() - t0, 2),
            "note": "Whisper-small CPU int8，仅作少读/跑飞的内容线索，不作音质或听感结论",
        }
        print("%-12s %.1fs  CER=%.4f  asr=%.0fs" %
              (blind, dur, out[blind]["cer"], out[blind]["asr_seconds"]),
              flush=True)
    json.dump(out, open(os.path.join(STATE, "asr_screen.json"), "w"),
              ensure_ascii=False, indent=1)
    print("-> asr_screen.json", flush=True)


if __name__ == "__main__":
    main()
