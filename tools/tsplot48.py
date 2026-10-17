# -*- coding: utf-8 -*-
"""tsplot48.py —— 画「我们 vs EL」的时序结构分布对比图（免对齐统计）。

输出两张直方图：发声段时长分布、静音段时长分布。
用来直观展示节奏层面我们和 EL 差在哪（比表格好懂）。

用法：python3 tsplot48.py --ours <音频...> --el <音频...> [--out png] [--title 前缀]
"""
import argparse
import subprocess
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 0.005


def load(p):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(SR), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    return x


def struct(p):
    x = load(p)
    n = int(SR * FRAME)
    m = len(x) // n * n
    db = 20 * np.log10(np.maximum(
        np.sqrt((x[:m].reshape(-1, n) ** 2).mean(axis=1)), 1e-12))
    v = db > db.mean() + 2.0
    mg = int(0.05 / FRAME)
    vv = v.copy()
    i = 0
    while i < len(vv):
        if not vv[i]:
            j = i
            while j + 1 < len(vv) and not vv[j + 1]:
                j += 1
            if (j - i + 1) <= mg:
                vv[i:j + 1] = True
            i = j + 1
        else:
            i += 1
    segs, gaps, i = [], [], 0
    while i < len(vv):
        if vv[i]:
            j = i
            while j + 1 < len(vv) and vv[j + 1]:
                j += 1
            segs.append((j - i + 1) * FRAME)
            i = j + 1
        else:
            j = i
            while j + 1 < len(vv) and not vv[j + 1]:
                j += 1
            gaps.append((j - i + 1) * FRAME)
            i = j + 1
    return segs, gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", nargs="+", required=True)
    ap.add_argument("--el", nargs="+", required=True)
    ap.add_argument("--out", default="ts_compare.png")
    ap.add_argument("--title", default="timing structure")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def collect(fs):
        s, g = [], []
        for f in fs:
            x, y = struct(f)
            s += x
            g += y
        return np.array(s), np.array(g)

    os_, og = collect(a.ours)
    es, eg = collect(a.el)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    # 发声段
    ax = axes[0]
    bins = np.linspace(0, 1.0, 61)
    ax.hist(os_, bins=bins, density=True, alpha=0.55, label="ours", color="#ff6b6b")
    ax.hist(es, bins=bins, density=True, alpha=0.55, label="EL", color="#4aa3ff")
    ax.axvline(np.median(os_), color="#ff6b6b", ls="--", lw=1.6)
    ax.axvline(np.median(es), color="#4aa3ff", ls="--", lw=1.6)
    ax.set_title("voiced segment duration (median ours %.3fs vs EL %.3fs)"
                 % (np.median(os_), np.median(es)), fontsize=11)
    ax.set_xlabel("s")
    ax.legend()
    ax.grid(alpha=0.2)

    # 静音段
    ax = axes[1]
    bins = np.linspace(0, 0.9, 61)
    ax.hist(og, bins=bins, density=True, alpha=0.55, label="ours", color="#ff6b6b")
    ax.hist(eg, bins=bins, density=True, alpha=0.55, label="EL", color="#4aa3ff")
    ax.axvline(0.4, color="#00ff88", ls=":", lw=1.8)
    ax.text(0.41, ax.get_ylim()[1] * 0.85, "0.4s = long gap",
            color="#00ff88", fontsize=9)
    ax.set_title("silence duration (long>=0.4s: ours %.1f%% vs EL %.1f%%)"
                 % ((og >= 0.4).mean() * 100, (eg >= 0.4).mean() * 100), fontsize=11)
    ax.set_xlabel("s")
    ax.legend()
    ax.grid(alpha=0.2)

    plt.suptitle("%s: ours vs EL (alignment-free distribution)" % a.title, y=1.02)
    plt.tight_layout()
    plt.savefig(a.out, dpi=100, bbox_inches="tight")
    print("saved", a.out)
    print("ours: seg n=%d med=%.3f | gap n=%d long>=0.4s %.1f%%"
          % (len(os_), np.median(os_), len(og), (og >= 0.4).mean() * 100))
    print("EL  : seg n=%d med=%.3f | gap n=%d long>=0.4s %.1f%%"
          % (len(es), np.median(es), len(eg), (eg >= 0.4).mean() * 100))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
