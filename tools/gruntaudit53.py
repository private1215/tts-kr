# -*- coding: utf-8 -*-
"""gruntaudit53.py —— 审计 gruntclean53.py 到底漏了多少「嗯」。

员工反馈（z26）：「整体有改善，但是没清理干净」。
→ 说明判据抓到了一部分、漏了一部分。本工具逐级放宽判据，定位漏在哪。

原判据（gruntclean53.py 默认）：
    时长 20~120ms，前后静音都 >0.15s

逐级放宽：
    A 原判据
    B 只放宽时长   20~250ms
    C 只放宽静音   静音 >0.08s
    D 两者都放宽   20~250ms + 静音 >0.08s
    E 再放宽静音   静音 >0.04s
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 0.005


def load16(p, sr=SR):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(sr), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def segments(x):
    n = int(FRAME * SR)
    nfr = len(x) // n
    rms = np.sqrt((x[:nfr * n].reshape(nfr, n) ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-12)
    voiced = db > np.percentile(db, 35)
    segs, i = [], 0
    while i < len(voiced):
        if voiced[i]:
            j = i
            while j < len(voiced) and voiced[j]:
                j += 1
            segs.append((i * FRAME, (j - i) * FRAME))
            i = j
        else:
            i += 1
    return segs


def find(x, lo_ms, hi_ms, sil):
    segs = segments(x)
    lo, hi = lo_ms / 1000.0, hi_ms / 1000.0
    out = []
    for k in range(1, len(segs) - 1):
        s, d = segs[k]
        if lo <= d <= hi:
            pe = segs[k - 1][0] + segs[k - 1][1]
            nx = segs[k + 1][0]
            if (s - pe) > sil and (nx - (s + d)) > sil:
                out.append((s, d))
    return out


VARIANTS = [
    ("A 原判据 20-120ms/sil0.15", 20, 120, 0.15),
    ("B 放宽时长 20-250ms/sil0.15", 20, 250, 0.15),
    ("C 放宽静音 20-120ms/sil0.08", 20, 120, 0.08),
    ("D 都放宽 20-250ms/sil0.08", 20, 250, 0.08),
    ("E 再放宽 20-250ms/sil0.04", 20, 250, 0.04),
]


def main():
    for p in sys.argv[1:]:
        x = load16(p)
        dur = len(x) / SR
        print("\n=== %s (%.1fs) ===" % (os.path.basename(p), dur))
        base = None
        for name, lo, hi, sil in VARIANTS:
            g = find(x, lo, hi, sil)
            if base is None:
                base = set(round(s, 2) for s, _ in g)
                extra = 0
            else:
                extra = len(set(round(s, 2) for s, _ in g) - base)
            print("  %-30s %4d 个 (%.2f/分钟)  比A多 %d"
                  % (name, len(g), len(g) / dur * 60, extra))
    return 0


if __name__ == "__main__":
    sys.exit(main())
