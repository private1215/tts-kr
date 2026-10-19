# -*- coding: utf-8 -*-
"""eqverify53.py —— 直接验证 EQ 是否真的改变了频谱（整段，不分帧）。

eqscan53.py 扫描发现：增益 0→5dB 后发声帧中频占比几乎不变（5.76→5.80），
怀疑 ffmpeg equalizer 的 w 参数没按预期生效。本脚本直接对比几个频段的
整段能量占比，确认 EQ 到底动没动。
"""
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
BANDS = [(0, 300), (300, 1000), (1000, 3000), (3000, 8000)]


def load16(p):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(SR), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    return x.astype(np.float64)


def bands(p):
    x = load16(p)
    sp = np.abs(np.fft.rfft(x)) ** 2
    fr = np.fft.rfftfreq(len(x), 1.0 / SR)
    tot = sp.sum() + 1e-12
    return [float(sp[(fr >= a) & (fr < b)].sum() / tot) * 100
            for a, b in BANDS]


def eq(p, g, w=1600):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p, "-af",
                    "equalizer=f=1800:t=q:w=%d:g=%g" % (w, g),
                    "-ar", str(SR), "-ac", "1", t], check=True)
    return t


def main():
    src = sys.argv[1]
    print("整段频段能量占比（%%）")
    print("%-14s %8s %8s %8s %8s" % ("", "<300", "300-1k", "1k-3k", "3k-8k"))
    b0 = bands(src)
    print("%-14s %8.2f %8.2f %8.2f %8.2f" % ("原始", *b0))
    for g in (3, 5, 8):
        b = bands(eq(src, g))
        print("EQ+%gdB(w=1600) %8.2f %8.2f %8.2f %8.2f" % (g, *b))
    # 换更宽/更窄的 w 再试一次
    for w in (300, 6000):
        b = bands(eq(src, 5, w))
        print("EQ+5dB(w=%d)%s %8.2f %8.2f %8.2f %8.2f"
              % (w, " " * max(0, 6 - len(str(w))), *b))
    return 0


if __name__ == "__main__":
    sys.exit(main())
