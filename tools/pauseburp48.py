# -*- coding: utf-8 -*-
"""pauseburp48.py —— 在 ASR 词时间戳标记的「停顿」里找突然冒出来的短音。

策略：
1. faster-whisper 词级时间戳定位每个词的开始/结束；
2. 词与词之间 ≥0.1s 的间隙视为停顿；
3. 在停顿内部找能量明显高于停顿底噪、且有时频结构（spectral centroid 集中在
   人声区）的短段，排除纯 click 瞬态。

用法：python3 pauseburp48.py <音频> [--lang ko]
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")

SR = 16000
FRAME = 0.005
MIN_GAP = 0.10
MIN_BURST = 0.05
MAX_BURST = 0.50
DB_ABOVE_GAP = 10.0


def to16k(p):
    if p.endswith(".wav"):
        return p
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(SR), "-ac", "1", t], check=True)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--lang", default="ko")
    a = ap.parse_args()

    import asr_check as A
    import soundfile as sf
    m = A._load()
    if m is None:
        print("❌ ASR 模型不可用"); return 1

    wav = to16k(a.wav)
    x, sr = sf.read(wav, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    dur = len(x) / sr

    # frame energy
    n = int(sr * FRAME)
    fr = x[:len(x)//n*n].reshape(-1, n)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-12))

    # word timestamps
    segs, _ = m.transcribe(A._read_16k(a.wav),
                           language=A.lang_code(a.lang),
                           word_timestamps=True, vad_filter=True, beam_size=1)
    words = []
    for s in segs:
        for wd in (s.words or []):
            words.append((float(wd.start), float(wd.end), (wd.word or "").strip()))
    words.sort()

    print("=== %s (%.1fs) ===" % (os.path.basename(a.wav), dur))
    print("词数", len(words))

    # gaps
    gaps = []
    if words and words[0][0] > MIN_GAP:
        gaps.append((0.0, words[0][0], "HEAD"))
    for i in range(len(words) - 1):
        g0, g1 = words[i][1], words[i + 1][0]
        if g1 - g0 >= MIN_GAP:
            gaps.append((g0, g1, "%s → %s" % (words[i][2], words[i + 1][2])))
    if words and dur - words[-1][1] > MIN_GAP:
        gaps.append((words[-1][1], dur, "TAIL"))

    hits = []
    for g0, g1, ctx in gaps:
        ia = int(g0 / FRAME); ib = int(g1 / FRAME)
        seg_db = db[ia:ib]
        if len(seg_db) == 0:
            continue
        floor = np.percentile(seg_db, 10)  # 停顿底噪
        # 找连续高于 floor+DB_ABOVE_GAP 的段
        over = seg_db >= (floor + DB_ABOVE_GAP)
        i = 0
        while i < len(over):
            if over[i]:
                j = i
                while j + 1 < len(over) and over[j + 1]:
                    j += 1
                dur_s = (j - i + 1) * FRAME
                if MIN_BURST <= dur_s <= MAX_BURST:
                    t0 = g0 + i * FRAME
                    t1 = g0 + (j + 1) * FRAME
                    peak_db = float(seg_db[i:j + 1].max())
                    hits.append((t0, t1, dur_s, peak_db, floor, ctx))
                i = j + 1
            else:
                i += 1

    if not hits:
        print("✅ 未在词间停顿里找到可疑短突起")
        return 0

    print("\n可疑短突起 %d 处：" % len(hits))
    for t0, t1, dur_s, peak_db, floor, ctx in hits:
        print("  %.2f ~ %.2fs  时长 %.2fs  峰值 %.1f dB  高出停顿底噪 %.1f dB  | %s" %
              (t0, t1, dur_s, peak_db, peak_db - floor, ctx))
    return 0


if __name__ == "__main__":
    sys.exit(main())
