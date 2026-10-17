# -*- coding: utf-8 -*-
"""tailgrunt53b.py —— 严格判据检测「句末尾音嗯」（浊音 + 可闻时长 + 实质能量）。

tailgrunt53.py 判据太松：把 5ms 噪声段和正常句末短段都算进去，
结果我们 141 个、EL 也 136 个，没有判别力。

本工具收紧为（员工听到的「嗯」必须同时满足）：
    · 紧跟长停顿（>= PAUSE_MIN）之前
    · 时长 TAIL_MS 范围（默认 50~250ms，短于听不出、长于是正常音节）
    · 峰值 > PEAK_DB（默认 -28dB，有实质能量）
    · **浊音特征**：过零率 < ZCR_MAX 且 300Hz 以下能量占比 > LOW_RATIO
      （「嗯」是鼻浊音，低频饱满、无高频摩擦）

再对比我们与 EL：找「我们有、EL 没有」的判别式。
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 0.005
PAUSE_MIN = 0.25
TAIL_MS = (50.0, 250.0)
PEAK_DB = -28.0
ZCR_MAX = 0.12
LOW_RATIO = 0.45


def load16(p, sr=SR):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(sr), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def voiced_quality(a):
    """浊音判定：过零率 + 低频占比。"""
    if len(a) < 40:
        return 1.0, 0.0
    zcr = float(np.mean(np.abs(np.diff(np.sign(a)))) / 2)
    sp = np.abs(np.fft.rfft(a * np.hanning(len(a)))) ** 2
    fr = np.fft.rfftfreq(len(a), 1.0 / SR)
    low = float(sp[fr < 300].sum() / (sp.sum() + 1e-12))
    return zcr, low


def analyse(x):
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
            segs.append((i * FRAME, (j - i) * FRAME, float(db[i:j].max())))
            i = j
        else:
            i += 1
    tails = []
    for k in range(1, len(segs)):
        s, d, pk = segs[k - 1]
        gap = segs[k][0] - (s + d)
        if gap < PAUSE_MIN:
            continue
        if not (TAIL_MS[0] <= d * 1000 <= TAIL_MS[1]):
            continue
        if pk < PEAK_DB:
            continue
        a = x[int(s * SR):int((s + d) * SR)]
        zcr, low = voiced_quality(a)
        if zcr < ZCR_MAX and low > LOW_RATIO:
            tails.append((s, d * 1000, pk, zcr, low))
    return segs, tails


def main():
    print("判据：长停顿(>=%.2fs)前紧贴的发声段，%d~%dms，peak>%.0fdB，"
          "浊音(zcr<%.2f 且 低频>%.2f)"
          % (PAUSE_MIN, TAIL_MS[0], TAIL_MS[1], PEAK_DB, ZCR_MAX, LOW_RATIO))
    print("%-26s %8s %10s %10s" % ("文件", "发声段", "尾音嗯", "次/分钟"))
    rows = {}
    for p in sys.argv[1:]:
        x = load16(p)
        segs, tails = analyse(x)
        dur = len(x) / SR
        rows[os.path.basename(p)] = (len(segs), len(tails), dur, tails)
        print("%-26s %8d %10d %10.2f"
              % (os.path.basename(p), len(segs), len(tails), len(tails) / dur * 60))
        for s, ms, pk, zcr, low in tails[:12]:
            print("      t=%7.2fs  %3.0fms  peak=%.1fdB  zcr=%.3f 低频=%.2f"
                  % (s, ms, pk, zcr, low))
    return 0


if __name__ == "__main__":
    sys.exit(main())
