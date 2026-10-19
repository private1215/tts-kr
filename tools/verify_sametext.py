# -*- coding: utf-8 -*-
"""verify_sametext.py —— 按 (文件, 起始秒, 时长) 转录，确认是否同文本。"""
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel


def read16k(p, start, dur):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(start),
                    "-t", str(dur), "-i", p,
                    "-ar", "16000", "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def main():
    m = WhisperModel("small", device="cpu", compute_type="int8",
                     local_files_only=True)
    jobs = [
        ("/mnt/tts/fb/z29_m_v4.wav", 0, 14),        # 我们 m_clean 的 80~94s
        ("/mnt/tts/data/cases/d172b8c4efb3/merged.wav", 78, 14),  # EL 同位置
        ("/mnt/tts/data/cases/d172b8c4efb3/merged.wav", 155, 14),  # EL 再往后一处
    ]
    for p, st, du in jobs:
        x = read16k(p, st, du)
        segs, info = m.transcribe(x, language="ko", vad_filter=False)
        txt = " ".join(s.text.strip() for s in segs)
        print("=" * 20)
        print("%s  @%.0fs+%.0fs" % (p, st, du))
        print(txt[:300])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
