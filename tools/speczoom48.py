# -*- coding: utf-8 -*-
"""speczoom48.py —— 把 click 候选时刻 ±0.4s 放大，看看到底是瞬态还是「嘎哈」短音。

用法：python3 speczoom48.py <音频> <时刻1,时刻2,...>
"""
import os
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

SR = 16000
WINDOW = 0.4


def load(p):
    if not p.endswith(".wav"):
        t = tempfile.mktemp(suffix=".wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                        "-ar", str(SR), "-ac", "1", t], check=True)
        p = t
    x, sr = sf.read(p, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def main():
    src = sys.argv[1]
    times = [float(t) for t in sys.argv[2].split(",")]
    out = "zoom_" + os.path.basename(src).replace(".mp3", "").replace(".wav", "") + ".png"
    x, sr = load(src)
    ncols = 4
    nrows = int(np.ceil(len(times) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6.2, nrows * 3.2))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    for idx, c in enumerate(times):
        ax = axes[idx // ncols, idx % ncols]
        t0 = max(0, c - WINDOW)
        t1 = min(len(x) / sr, c + WINDOW)
        seg = x[int(t0 * sr):int(t1 * sr)]
        ax.specgram(seg, NFFT=512, Fs=sr, noverlap=384,
                    cmap="magma", vmin=-90, vmax=-10)
        ax.set_ylim(0, 8000)
        ax.axvline(c - t0, color="#00ff88", lw=1.5, ls="--")
        ax.set_title("t=%.2fs" % c, fontsize=10)
        ax.set_xlabel("s")
        ax.set_ylabel("Hz")
    for idx in range(len(times), nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")
    plt.suptitle("click/burst zoom ±%.1fs — %s" % (WINDOW, os.path.basename(src)), y=1.02)
    plt.tight_layout()
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print("saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
