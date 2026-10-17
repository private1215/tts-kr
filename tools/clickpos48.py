# -*- coding: utf-8 -*-
"""clickpos48.py —— click 定位诊断：这些硬切在**哪里**？

若 click 呈**周期性**（每 ~N 秒一次）→ 是长文本**分块拼接**的块边界硬切，
修法是在块边界做交叉淡化，与音频内容无关。
若随机散布且集中在发声中 → 是合成器本身产物，只能换引擎或重训练。

用法：python3 clickpos48.py <wav/mp3> [更多...]
"""
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
THR = 0.30


def load(p):
    # 🔴🔴 必须对**所有**格式都降到 16k，不能只看扩展名是不是 .wav！
    # 旧版 `if not p.endswith(".wav")` 会让 .wav 输入按原采样率算：
    # 案例库 EL 的 merged.wav 是 24k/48k，算出来的 click 次数只有 16k 口径的
    # 1/13（实测 10 vs 132），据此得出「我们是 EL 的 26 倍」——**错得离谱**。
    # 真实差距（16k 统一口径）是 2.0 倍，GS 女声反而比 EL 好 40%。
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(SR), "-ac", "1", t], check=True)
    x, sr = sf.read(t, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def analyze(path):
    x, sr = load(path)
    x = x / (np.abs(x).max() + 1e-12) * 0.70     # 🔴 必须峰值归一
    d = np.abs(np.diff(x))
    hits = np.where(d >= THR)[0]
    if len(hits) == 0:
        print("%-30s click 0" % path.split("/")[-1])
        return
    t = hits / sr
    gaps = np.diff(t)
    # 只统计「够远」的间隔，避开同一瞬态的连续采样
    g = gaps[gaps > 0.05]
    # 是否在发声中：click 前后 100ms 的 RMS vs 全篇 RMS
    n = int(sr * 0.1)
    def rms_at(i):
        a = max(0, i - n); b = min(len(x), i + n)
        return float(np.sqrt((x[a:b] ** 2).mean()))
    ctx = np.array([rms_at(i) for i in hits[:400]])
    whole = float(np.sqrt((x ** 2).mean()))
    voiced = float((ctx > whole * 0.3).mean() * 100)

    print("\n=== %s ===" % path.split("/")[-1])
    print("  click 次数 %d   密度 %.3f%%   时长 %.1fs" % (
        len(hits), len(hits) / len(x) * 100, len(x) / sr))
    print("  发声中占比 %.0f%%（前后 100ms RMS > 全篇 0.3 倍）" % voiced)
    if len(g) > 5:
        print("  间隔(>50ms)：中位 %.2fs  最小 %.2fs  最大 %.2fs" % (
            np.median(g), g.min(), g.max()))
        # 🔴 周期性检验：把间隔除以中位间隔，看是否集中在 1.0 附近
        r = np.sort(g) / np.median(g)
        near = float(((r > 0.8) & (r < 1.25)).mean() * 100)
        print("  周期性：%.0f%% 的间隔落在中位 ±25%% 内"
              "（>60%% 说明是固定周期的块边界）" % near)
        # 前 12 个 click 的绝对时间
        print("  前 12 个 click 时刻(s)：%s" %
              " ".join("%.1f" % v for v in t[:12]))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
