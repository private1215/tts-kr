#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""尾部 vs 中段：我们 vs EL 的「韵律/节奏」专项对比

## 为什么要做
z6 轮（3 人全有效，控制题全对）：
- 发客户基线 3/3 全 EL
- 中段 vs EL：③听不出区别 / ②EL更像 / ④两段都像机器 → **中段已接近**
- **尾部 vs EL：3/3 全票「EL 更像真人」** ← 尾部是唯一全员一致的短板

但 z36 的逐窗客观数据里尾部并不差：
音色 前0.779/中0.770/**后0.793**、F0 189.7、发声占比 0.564、平坦度 0.0442
→ 说明**尾部问题不在音色/频段，可能在韵律/节奏**（F0 起伏、停顿、能量变化）。

## 量什么（都是听感"像不像真人"相关的老指标）
- F0：中位数 / 标准差 / p10~p90 跨度（起伏越大越自然）
- 停顿：静音段（<阈值）个数、平均时长、最长
- 能量：帧电平标准差（dB）
- 频谱质心、音色相似度（campplus）
对比：我们的中段 vs EL 中段、我们的尾部 vs EL 尾部。
"""
import os
import subprocess
import sys

import numpy as np
import librosa
import soundfile as sf
import torch

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/tools")
import onnxruntime  # noqa: E402
import spkchk  # noqa: E402

TMP = "/mnt/tts/diag/_tail"
os.makedirs(TMP, exist_ok=True)

OURS = "/mnt/tts/output/longprobe-z36-6cdd/longprobe-z36-6cdd-1.mp3"
EL_S = "/mnt/tts/data/cases/6cdd25bc75f2/merged.wav"
REF = "/mnt/tts/data/voices/EL_male.wav"

# (标签, 源, 起点, 时长)  对齐：我们 496s / EL 509s 同文案，尾部偏差按 2.6% 折算
SEGS = [
    ("我们的 中段", OURS, 240.0, 30.0),
    ("EL    中段", EL_S, 245.0, 30.0),
    ("我们的 尾部", OURS, 440.0, 20.0),
    ("EL    尾部", EL_S, 448.0, 20.0),
    ("我们的 开头", OURS, 0.0, 30.0),
    ("EL    开头", EL_S, 0.0, 30.0),
]

sess = onnxruntime.InferenceSession(spkchk.CAMPPLUS,
                                    providers=["CPUExecutionProvider"])
E_REF = spkchk.embed(sess, spkchk.read16(REF))


def cut(src, s0, dur, tag):
    out = os.path.join(TMP, "%s.wav" % tag)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ss", "%.1f" % s0,
                    "-t", "%.1f" % dur, "-ac", "1", "-ar", "16000", out],
                   check=True)
    y, _ = sf.read(out, dtype="float32")
    return y


def metrics(y):
    sr = 16000
    o = {}
    f0, vo, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr)
    v = f0[np.isfinite(f0)]
    if len(v):
        o["f0_med"] = float(np.median(v))
        o["f0_std"] = float(np.std(v))
        o["f0_span"] = float(np.percentile(v, 90) - np.percentile(v, 10))
    else:
        o.update(f0_med=float("nan"), f0_std=float("nan"), f0_span=float("nan"))
    hop = int(0.01 * sr)
    n = len(y) // hop
    env = np.sqrt(np.array([(y[i * hop:(i + 1) * hop] ** 2).mean() for i in range(n)]))
    db = 20 * np.log10(env + 1e-12)
    thr = np.percentile(db, 30)          # 相对阈值（与 probe_g 的 voiced 口径同源）
    sil = db < thr
    # 停顿段统计
    d = np.diff(sil.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if sil[0]:
        starts = np.r_[0, starts]
    if sil[-1]:
        ends = np.r_[ends, len(sil)]
    m = min(len(starts), len(ends))
    if m:
        lens = (ends[:m] - starts[:m]) * 0.01
        o["pause_n"] = int((lens > 0.05).sum())
        o["pause_avg"] = float(lens[lens > 0.05].mean()) if (lens > 0.05).any() else 0.0
        o["pause_max"] = float(lens.max())
    else:
        o.update(pause_n=0, pause_avg=0.0, pause_max=0.0)
    o["db_std"] = float(np.std(db[db > thr]))          # 有声段的能量起伏
    o["centroid"] = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    o["sim"] = float(np.dot(E_REF, spkchk.embed(
        sess, torch.from_numpy(np.ascontiguousarray(y)).unsqueeze(0))))
    o["dur"] = len(y) / float(sr)
    return o


def main():
    print("%-12s %6s %7s %7s %7s %6s %7s %7s %7s %7s"
          % ("段", "时长", "音色", "F0中位", "F0std", "F0跨度", "停顿数", "停顿均",
             "dBstd", "质心"))
    rows = {}
    for i, (tag, src, s0, dur) in enumerate(SEGS):
        y = cut(src, s0, dur, "s%d" % i)
        o = metrics(y)
        rows[tag] = o
        print("%-12s %6.1f %7.4f %7.1f %7.2f %7.1f %6d %7.2f %7.2f %7.0f"
              % (tag, o["dur"], o["sim"], o["f0_med"], o["f0_std"], o["f0_span"],
                 o["pause_n"], o["pause_avg"], o["db_std"], o["centroid"]))
    print("\n=== 我们 / EL 差值（差得多的就是短板）===")
    for where in ("中段", "尾部", "开头"):
        a, b = rows["我们的 " + where], rows["EL    " + where]
        print("  %s：音色 %+.4f  F0std %+.2f  F0跨度 %+.1f  停顿数 %+d  "
              "停顿均 %+.2fs  dBstd %+.2f  质心 %+.0f"
              % (where, a["sim"] - b["sim"], a["f0_std"] - b["f0_std"],
                 a["f0_span"] - b["f0_span"], a["pause_n"] - b["pause_n"],
                 a["pause_avg"] - b["pause_avg"], a["db_std"] - b["db_std"],
                 a["centroid"] - b["centroid"]))
    print("\nTAIL_DONE")


if __name__ == "__main__":
    main()
