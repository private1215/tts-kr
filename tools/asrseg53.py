# -*- coding: utf-8 -*-
"""asrseg53.py —— 长音频分段 ASR（避免 whisper 长输入退化成循环复读）。

背景：全篇一次喂给 faster-whisper small（vad_filter=False）时，459s 的 mp3
会吐出大量重复的胡话（"육담카살이, 그만둬야마대" 循环），而同样的 453s raw
wav 却完全正常 —— 这是 whisper 长输入的已知退化，不代表音频真的坏了。
本工具按 30s 窗切开逐段转写再拼接，并用**同样口径**转多个文件做对照，
避免「不同文件的 ASR 不可比」这种假结论。

用法:
  python -u tools/asrseg53.py A.wav [B.mp3 ...] [--ref ref.txt]
                              [--win 30] [--full] [--lang ko]
输出: 每个文件的转写文本 + 与 --ref 的文本相似度（不给 ref 时与第一个文件比）。
"""
import argparse
import difflib
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf


def to16k(path, d):
    out = os.path.join(d, "x.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "16000", "-ac", "1", out], check=True)
    return out


def read16k(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def trans_chunks(m, x, sr, win=30.0):
    n = int(win * sr)
    texts = []
    for i in range(0, len(x), n):
        seg = x[i:i + n]
        if len(seg) < sr * 0.5:
            continue
        segs, _ = m.transcribe(seg, language=LANG, vad_filter=False)
        texts.append(" ".join(s.text.strip() for s in segs))
    return " ".join(texts)


LANG = "ko"


def main():
    global LANG
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--win", type=float, default=30.0)
    ap.add_argument("--full", action="store_true",
                    help="额外跑一遍整篇一次转写（复现退化用）")
    ap.add_argument("--lang", default="ko")
    a = ap.parse_args()
    LANG = a.lang

    from faster_whisper import WhisperModel
    m = WhisperModel("small", device="cpu", compute_type="int8",
                     local_files_only=True)

    texts = {}
    ref = None
    if a.ref:
        with open(a.ref, encoding="utf-8") as f:
            ref = f.read()
    d = tempfile.mkdtemp(prefix="asrseg_")
    for p in a.files:
        w = to16k(p, d) if not p.endswith(".wav") else p
        x, sr = read16k(w)
        t = trans_chunks(m, x, sr, a.win)
        texts[p] = t
        print("=" * 20, "%s  (%.1fs)" % (p, len(x) / float(sr)))
        print(t[:4000])
        if a.full:
            segs, _ = m.transcribe(x, language=LANG, vad_filter=False)
            tf = " ".join(s.text.strip() for s in segs)
            print("-" * 10, "整篇一次转写:")
            print(tf[:4000])
            texts[p + " [full]"] = tf
        sys.stdout.flush()

    base = ref if ref is not None else list(texts.values())[0]
    bl = "REF" if ref is not None else os.path.basename(a.files[0])
    print("=" * 20, "相似度（基准 = %s）" % bl)
    for k, v in texts.items():
        r = difflib.SequenceMatcher(None, base, v).ratio()
        print("  %.4f  %s" % (r, k))


if __name__ == "__main__":
    main()
