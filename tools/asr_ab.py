# -*- coding: utf-8 -*-
"""asr_ab.py —— 转录清理前后同一片段，对比文本相似度（验证没削掉字）。"""
import difflib

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel


def read16k(p):
    x, _ = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def main():
    import sys
    m = WhisperModel("small", device="cpu", compute_type="int8",
                     local_files_only=True)
    texts = []
    pair = sys.argv[1:3] if len(sys.argv) >= 3 else [
        "/mnt/tts/diag/pre_100_120.wav",
        "/mnt/tts/diag/post_100_120.wav"]
    for p in pair:
        x = read16k(p)
        segs, _ = m.transcribe(x, language="ko", vad_filter=False)
        t = " ".join(s.text.strip() for s in segs)
        texts.append(t)
        print("=" * 15, p)
        print(t)
    a, b = texts
    sm = difflib.SequenceMatcher(None, a, b)
    print("=" * 15)
    print("文本相似度 = %.4f（1.0=完全一致）" % sm.ratio())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
