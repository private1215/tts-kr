#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fback53b.py —— 按员工报的时间点逐段取证：那里到底有没有「嗯/啊/额」？

员工报的时间点（秒）：
男 0(开头异声) / 5.6(电磁音) / 8(0.1s没音) / 14(额) / 27 42 57 71 84 100 113 126 139(嗯)
女 0(开头异声) / 12(额) / 27 43 55 68 81 95 110 122 137 152 164(嗯/啊/额/叮)

方法：每个时间点取 [t-0.8, t+1.2] 一小段，
① 打印 10ms 帧能量包络（看是语音/静音/瞬态）
② ASR 转写这一小段（听到什么就打印什么，不猜）
③ 顺便给整段（0-200s）的带时间 ASR，定位真正的「额外音节」
"""
import subprocess
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")
from asr_check import normalize  # noqa: E402
import asr_check  # noqa: E402

SR = 16000

TARGETS = [
    ("男 202610210738", "/mnt/tts/output/202610210738/202610210738-1.mp3",
     [0.0, 5.6, 8.0, 14.0, 27.0, 42.0, 57.0, 71.0, 84.0, 100.0, 113.0,
      126.0, 139.0]),
    ("女 202610210939", "/mnt/tts/output/202610210939/202610210939-1.mp3",
     [0.0, 12.0, 27.0, 43.0, 55.0, 68.0, 81.0, 95.0, 110.0, 122.0, 137.0,
      152.0, 164.0]),
]


def extract(mp3, t0, dur, out="/tmp/fb53b.wav"):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", mp3,
                    "-ss", "%.2f" % max(t0, 0.0), "-ar", str(SR), "-ac", "1",
                    "-t", "%.2f" % dur, out], check=True)
    return sf.read(out, dtype="float32")[0]


def env(y, hop_ms=10.0):
    hop = int(hop_ms / 1000 * SR)
    n = max(1, len(y) // hop)
    db = 20 * np.log10(np.array(
        [np.sqrt(np.mean(y[i * hop:(i + 1) * hop] ** 2) + 1e-12)
         for i in range(n)]) + 1e-12)
    return db, hop / SR


def bar(db):
    """把 dB 序列画成粗 ASCII 条（-70 以下=空格）。"""
    s = ""
    for d in db:
        s += " .:-=+*#@"[min(8, max(0, int((d + 70) / 9)))]
    return s


def main():
    m = asr_check._load()
    for tag, path, times in TARGETS:
        print("=" * 74)
        print("%s" % tag)
        for t in times:
            y = extract(path, t - 0.8, 2.0)
            db, dt = env(y)
            txt = ""
            try:
                segs, _ = m.transcribe(y, language="ko", vad_filter=False,
                                       beam_size=1)
                txt = " ".join(s.text.strip() for s in segs)
            except Exception as e:
                txt = "(失败 %r)" % repr(e)[:40]
            print("\n  %6.1fs  %s" % (t, bar(db)))
            print("          ASR=%r" % txt[:60])
            print("          帧dB 最小%.0f 最大%.0f"
                  % (db.min(), db.max()))
        print()


if __name__ == "__main__":
    sys.exit(main())
