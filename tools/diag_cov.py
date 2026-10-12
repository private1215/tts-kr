# -*- coding: utf-8 -*-
"""诊断：为什么「参照原声」的 ASR 覆盖率只有 0.046。

ref_text 本来就是从这段音频 ASR 转出来的，覆盖率理应接近 1。
若不是，说明**评分用的链路和生成文案用的链路不同源**，分数不可信。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from app import asr_check as ASR

OUT = "/mnt/tts/projects/gpt-sovits/out"
TMP = "/tmp/_diag16k.wav"


def to16k(path):
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "16000", "-ac", "1", TMP], check=True)
    return TMP


def main():
    for slot in ("000", "150", "300"):
        for spk in ("sola", "yohan"):
            wav = os.path.join(OUT, "probe_%s_%s.wav" % (spk, slot))
            txt = os.path.join(OUT, "text_%s_%s.txt" % (spk, slot))
            if not (os.path.exists(wav) and os.path.exists(txt)):
                continue
            ref = ASR.normalize(open(txt, encoding="utf-8").read())
            try:
                hyp = ASR.normalize(ASR.transcribe(to16k(wav), "ko"))
            except Exception as e:
                print("%s/%s 转录异常 %r" % (spk, slot, e))
                continue
            print("%s/%s  ref=%d字  hyp=%d字  hyp前60=%r"
                  % (spk, slot, len(ref), len(hyp), hyp[:60]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
