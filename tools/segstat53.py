# -*- coding: utf-8 -*-
"""segstat53.py —— 分段能量/发声占比统计，用来判断某段是不是「特效/片头」。
用法：python3 segstat53.py <wav> [wav2 ...]
"""
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
EDGES = [(0, 30), (30, 60), (60, 120), (120, 300), (300, 600), (600, 900)]


def load16(p):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(SR), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    return x.astype(np.float64)


def main():
    for p in sys.argv[1:]:
        x = load16(p)
        dur = len(x) / SR
        n = int(0.005 * SR)
        nfr = len(x) // n
        rms = np.sqrt((x[:nfr * n].reshape(nfr, n) ** 2).mean(axis=1))
        db = 20 * np.log10(rms + 1e-12)
        print("=== %s  %.1fs  全篇 中位%.1fdB p95 %.1f max %.1f"
              % (p.split("/")[-1], dur, np.median(db),
                 np.percentile(db, 95), db.max()))
        for a, b in EDGES:
            if a >= dur:
                break
            b = min(b, dur)
            s = db[int(a / 0.005):int(b / 0.005)]
            if not len(s):
                continue
            print("   %3d-%3ds  发声占比%.2f  中位%6.1fdB  p95 %6.1f  max %6.1f"
                  % (a, b, (s > -40).mean(), np.median(s),
                     np.percentile(s, 95), s.max()))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
