#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""长篇跨窗一致性：我们 vs EL（2026-10-15 第31轮）

## 为什么这是验收的真正尺子
用户的原话：「短文案语音肯定都漂亮，长文案语音稳定才是实力。」
静态指标已经追平（Σz² 27.1 < EL 中位 59.6），但**长篇里各段是否一致**
才是员工听得出「像不像真人」的地方。

历史基线（整改前，记忆里的最严重项）：
| 跨窗极差% | 我们 | EL |
|---|---|---|
| 4k~8k | **94.9%** | 31.9% |
| 150~300 | 31.6% | 11.9% |
| 跨窗 Σz² 中位 | 783 | 50 |

本轮把「两次 atempo 揉两遍」合并成一次后，需要复查这些数字有没有跟着改善。

## 口径
- 统一 16k 单声道
- 30 秒一窗，**跳过首尾各 5%**（避开开场与收尾）
- 每窗算：7 个频段占比、平坦度、发声占比、F0 中位
- 报：极差%（(max−min)/median）与 CV%

## 用法
    ./venv/bin/python tools/consistency.py
"""
import json
import os
import subprocess

import numpy as np

SR = 16000
WIN = 30.0
OUT = "/mnt/tts/diag/muffle/consistency.json"

SRCS = [
    ("我们 r31(融合)", "/mnt/tts/output/longprobe-r31-6cdd/"
                       "longprobe-r31-6cdd-1.mp3"),
    ("我们 r30fix(旧)", "/mnt/tts/output/longprobe-r30fix-6cdd-2/"
                        "longprobe-r30fix-6cdd-1.mp3"),
    ("EL 标杆", "/mnt/tts/diag/muffle/el24.wav"),
]

BANDS = [(60, 150), (150, 300), (300, 500), (500, 1000),
         (1000, 2000), (2000, 4000), (4000, 8000)]

try:
    import librosa
except Exception:
    librosa = None


def load16(p):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-ac", "1",
                          "-ar", str(SR), "-f", "f32le", "-"],
                         stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def win_metrics(y):
    n, hop = 2048, 512
    nfr = (len(y) - n) // hop
    idx = np.arange(nfr)[:, None] * hop + np.arange(n)[None, :]
    fr = y[idx] * np.hanning(n)[None, :]
    sp = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1.0 / SR)
    tot = sp.sum(axis=1)
    out = {}
    for lo, hi in BANDS:
        m = (freq_mask(f, lo, hi))
        out["b%d_%d" % (lo, hi)] = float(sp[:, m].sum() / tot.sum() * 100)
    # 发声占比：能量高于该窗最大值的 -40dB 的帧占比
    e = 10 * np.log10(tot + 1e-20)
    out["voiced"] = float((e > (e.max() - 40)).mean())
    # 平坦度
    if librosa is not None:
        out["flat"] = float(librosa.feature.spectral_flatness(
            y=y, hop_length=256)[0].mean())
    else:
        out["flat"] = float("nan")
    # F0
    try:
        f0, _, _ = librosa.pyin(y, fmin=60, fmax=400, sr=SR)
        out["f0"] = float(np.nanmedian(f0)) if f0 is not None else float("nan")
    except Exception:
        out["f0"] = float("nan")
    return out


def freq_mask(f, lo, hi):
    return (f >= lo) & (f < hi)


def analyse(path):
    y = load16(path)
    n = int(WIN * SR)
    skip = int(len(y) * 0.05)
    segs = []
    s = skip
    while s + n <= len(y) - skip:
        segs.append(y[s:s + n])
        s += n
    rows = [win_metrics(sg) for sg in segs]
    keys = [k for k in rows[0]]
    res = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        v = v[~np.isnan(v)]
        if len(v) < 2:
            continue
        med = float(np.median(v))
        rng = float((v.max() - v.min()) / abs(med) * 100) if med else float("nan")
        cv = float(v.std() / abs(med) * 100) if med else float("nan")
        res[k] = {"med": med, "min": float(v.min()), "max": float(v.max()),
                  "range_pct": rng, "cv_pct": cv}
    return {"path": path, "n_win": len(rows), "dur": len(y) / SR, "m": res}


def main():
    out = []
    for tag, p in SRCS:
        if not os.path.exists(p):
            print("跳过（缺）:", p)
            continue
        r = analyse(p)
        r["tag"] = tag
        out.append(r)
        print("\n=== %s  (%.0fs, %d 窗) ===" % (tag, r["dur"], r["n_win"]))

    keys = ["b150_300", "b4000_8000", "flat", "voiced", "f0"]
    print("\n%-16s %s" % ("信号", "  ".join("%18s" % k for k in keys)))
    for r in out:
        cells = []
        for k in keys:
            m = r["m"].get(k)
            cells.append("%8.4f/%5.1f%%" % (m["med"], m["range_pct"]) if m
                         else " " * 18)
        print("%-16s %s" % (r["tag"], "  ".join("%18s" % c for c in cells)))
    print("\n（格式：中位 / 跨窗极差%）")

    if len(out) >= 2:
        print("\n=== 极差% 对比（越小越稳） ===")
        print("%-16s %s" % ("信号", "  ".join("%12s" % k for k in keys)))
        for r in out:
            print("%-16s %s" % (r["tag"], "  ".join(
                "%12s" % ("%.1f%%" % r["m"][k]["range_pct"] if k in r["m"]
                          else "-") for k in keys)))

    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("\n→ " + OUT)


if __name__ == "__main__":
    main()
