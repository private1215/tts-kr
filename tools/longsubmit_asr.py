# -*- coding: utf-8 -*-
"""整篇 ASR 内容筛查（Whisper-small / CPU / int8）。

⚠️ 定位：只作为「有没有少读、有没有跑飞」的**内容线索**，
不作音色 / 听感结论，更不能当作方案优劣的判据（用户明令）。
"""
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
from app.asr_check import cer, transcribe  # noqa: E402

STATE = "/mnt/tts/diag/longsubmit-20261013"
CASES = "/mnt/tts/data/cases"


def wavize(src, dst):
    """统一转成 24k 单声道 wav，便于 ffprobe/听感核对一致。"""
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src,
                    "-ac", "1", "-ar", "24000", dst], check=True)
    return dst


def main(targets):
    out = {}
    for name, (wav, cid) in targets.items():
        ref = open(os.path.join(CASES, cid, "full.txt"), encoding="utf-8").read()
        t0 = time.time()
        hyp = transcribe(wav, "ko")
        out[name] = {
            "wav": wav, "case_id": cid,
            "ref_chars": len(ref),
            "hyp_chars": len(hyp),
            "hyp_head": hyp[:120],
            "cer": cer(ref, hyp),
            "asr_seconds": round(time.time() - t0, 2),
            "note": "Whisper-small CPU int8，仅作少读/跑飞的内容线索，不作音质或听感结论",
        }
        print("%-10s CER=%.4f  asr=%.0fs  hyp=%d字" %
              (name, out[name]["cer"], out[name]["asr_seconds"],
               len(hyp)), flush=True)
    json.dump(out, open(os.path.join(STATE, "asr_screen.json"), "w"),
              ensure_ascii=False, indent=1)
    print("-> asr_screen.json")


if __name__ == "__main__":
    spec = json.loads(sys.argv[1])
    main(spec)
