# -*- coding: utf-8 -*-
"""posspec53.py —— 把员工标注位置的音频画成「波形 + 频谱 + 分段标记」三联图。

## 为什么

z28 员工给了「嗯」的**位置范围**（不是精确秒数也不是精确时长）：
    男 v4（片段起点80s）：5s / 21s / 49s / 1分03s  -> 绝对 85 / 101 / 129 / 143
    女 v3（片段起点340s）：10s / 23s / 34s / 47~50s / 1分 / 1分08s
                          -> 绝对 350 / 363 / 374 / 387 / 400 / 408

三轮猜判据都失败（v1~v4），且声学扫描显示「我们 vs EL」比值≈1
（`gruntscan53.py`），说明**按特征筛不出「嗯」**。
→ 不再猜，直接**看图定位**。

🔴 员工说的「0.001~0.05 秒」是**感觉时长**，不能当判据用（用户明确指出）。
   只信位置。

## 用法
    python3 posspec53.py <wav> <pos1> <pos2> ... [--win 0.5] [-o out.png]

图中竖线 = 标注位置；红色区间 = 判为发声的段（门槛 -45dB）。
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 0.005
VTH = -45.0


def load16(p, sr=SR):
    if p.lower().endswith(".wav"):
        x, s = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if s != sr:
            t = tempfile.mktemp(suffix=".wav")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                            "-ar", str(sr), "-ac", "1", t], check=True)
            x, _ = sf.read(t, dtype="float32")
            os.unlink(t)
        return x
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(sr), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    return x


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("pos", nargs="+", type=float)
    ap.add_argument("--win", type=float, default=0.5)
    ap.add_argument("--vmin", type=float, default=-85)
    ap.add_argument("--vmax", type=float, default=-15)
    ap.add_argument("-o", default="pos.png")
    a = ap.parse_args()

    x = load16(a.wav)
    n = len(x)
    nfr = n // int(FRAME * SR)
    rms = np.sqrt((x[:nfr * int(FRAME * SR)].reshape(nfr, -1) ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-12)

    npos = len(a.pos)
    fig, axes = plt.subplots(npos * 2, 1, figsize=(16, 2.0 * npos * 2))
    if npos == 1:
        axes = [axes]
    for i, pos in enumerate(a.pos):
        s0 = max(0.0, pos - a.win)
        s1 = min(n / SR, pos + a.win)
        i0, i1 = int(s0 * SR), int(s1 * SR)
        seg = x[i0:i1]
        t = np.arange(len(seg)) / SR + s0

        ax1 = axes[i * 2]
        ax1.plot(t, seg, lw=0.4, color="#1f77b4")
        ax1.axvline(pos, color="r", lw=1.2)
        ax1.set_ylabel("amp")
        ax1.set_title("%s  标注 %.0fs  (%.1f~%.1fs)"
                      % (os.path.basename(a.wav), pos, s0, s1), fontsize=10)
        ax1.set_ylim(-0.6, 0.6)
        ax1.grid(alpha=0.2)

        ax2 = axes[i * 2 + 1]
        P, freqs, bins, im = ax2.specgram(
            seg, NFFT=512, Fs=SR, noverlap=384,
            cmap="magma", vmin=-85, vmax=-15)
        im.remove()
        # 重建为绝对秒坐标的 pcolormesh
        S = 10 * np.log10(P + 1e-12)
        ax2.pcolormesh(bins + s0, freqs, S, shading="auto",
                       cmap="magma", vmin=a.vmin, vmax=a.vmax)
        ax2.set_xlim(s0, s1)
        ax2.axvline(pos, color="cyan", lw=1.2)
        ax2.set_ylim(0, 4000)
        ax2.set_ylabel("Hz")
        # 叠加 5ms 帧 dB 包络（右轴）
        f0 = int(s0 / FRAME)
        f1 = int(s1 / FRAME)
        tt = np.arange(f0, min(f1, len(db))) * FRAME
        ax2b = ax2.twinx()
        ax2b.plot(tt, db[f0:min(f1, len(db))], color="lime", lw=0.8, alpha=0.8)
        ax2b.axhline(VTH, color="lime", lw=0.6, ls="--", alpha=0.6)
        ax2b.set_ylabel("dB", color="lime")
        ax2b.set_ylim(-90, -10)
    axes[-1].set_xlabel("time (s)")
    plt.tight_layout()
    plt.savefig(a.o, dpi=110)
    print("-> %s" % a.o)
    return 0


if __name__ == "__main__":
    sys.exit(main())
