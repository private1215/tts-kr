#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第35轮反馈驱动的定位：长篇「逐窗漂移」分析

员工新反馈（3 人，控制题全对 = 有效）：
- 我们的 4 段（0 / 2:30 / 5:00 / 7:20）评级：均分 2.67 / 2.67 / 2.0 / **3.00**
  → **7:20（近结尾）3/3 全票「机器感很明显」，是全篇唯一全员一致的段落**
- EL 隐藏对照：①/②/①（均分 1.33）→ 标尺没失灵，差距是真的
- 发客户基线仍 3/3 全 EL

假设：**越到后面越电子**（长程漂移），前几轮只测前 90 秒，尾部从未单独测过。
本脚本按 20s 窗逐窗量：音色相似度 / F0 / 平坦度 / 发声占比 / 频段，
并同时跑 EL 案例作对照（EL 自己漂不漂 = 判断这是不是我们特有的问题）。

口径：campplus(spkchk.embed, 16k, nwin=12) 对 EL_male；
     平坦度 = librosa.feature.spectral_flatness(hop=256) 均值（与 probe_g 同口径）；
     频段 = 24k STFT n_fft2048 hop512 能量占比（与 probe_g 同口径）。
"""
import os
import subprocess
import sys

import numpy as np
import librosa
import soundfile as sf
import torch

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/tools")
import onnxruntime
import spkchk

WIN = float(os.environ.get("DRIFT_WIN", "20"))
TMP = "/mnt/tts/diag/_drift"
os.makedirs(TMP, exist_ok=True)

REF = os.environ.get("DRIFT_REF", "/mnt/tts/data/voices/EL_male.wav")
OURS = "/mnt/tts/output/longprobe-z32-6cdd/longprobe-z32-6cdd-1.mp3"
EL = "/mnt/tts/data/cases/6cdd25bc75f2/merged.wav"

sess = onnxruntime.InferenceSession(spkchk.CAMPPLUS,
                                    providers=["CPUExecutionProvider"])
E_REF = spkchk.embed(sess, spkchk.read16(REF))


def conv(src, sr, tag):
    out = os.path.join(TMP, "%s_%d.wav" % (tag, sr))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-ar", str(sr), "-ac", "1", out], check=True)
    y, _ = sf.read(out, dtype="float32")
    return y


def bands(y24):
    S = np.abs(librosa.stft(y24, n_fft=2048, hop_length=512)) ** 2
    f = librosa.fft_frequencies(sr=24000, n_fft=2048)
    tot = S.sum() + 1e-12
    r = {}
    for lo, hi in ((150, 300), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)):
        r["b%d_%d" % (lo, hi)] = float(S[(f >= lo) & (f < hi)].sum() / tot)
    return r


def win_metrics(y16, y24):
    o = {}
    o["sim"] = float(np.dot(
        E_REF, spkchk.embed(sess, torch.from_numpy(np.ascontiguousarray(y16)).unsqueeze(0))))
    f0, vo, _ = librosa.pyin(y16, fmin=60, fmax=400, sr=16000)
    v = f0[np.isfinite(f0)]
    o["f0"] = float(np.median(v)) if len(v) else float("nan")
    o["flat"] = float(librosa.feature.spectral_flatness(y=y16, hop_length=256)[0].mean())
    hop = int(0.01 * 16000)
    n = len(y16) // hop
    env = np.sqrt(np.array([(y16[i * hop:(i + 1) * hop] ** 2).mean()
                            for i in range(n)]))
    o["voiced"] = float((env > np.percentile(env, 30) * 2.2).mean())
    o.update(bands(y24))
    return o


def run(tag, src):
    y16, y24 = conv(src, 16000, tag), conv(src, 24000, tag)
    dur = min(len(y16) / 16000.0, len(y24) / 24000.0)
    rows = []
    s = 0.0
    while s + WIN <= dur:
        a, b = int(s * 16000), int((s + WIN) * 16000)
        c, d = int(s * 24000), int((s + WIN) * 24000)
        m = win_metrics(y16[a:b], y24[c:d])
        m["t"] = s
        rows.append(m)
        s += WIN
    return rows, dur


def show(tag, rows, dur):
    print("\n===== %s（%.1fs，%d 个 %.0fs 窗）=====" % (tag, dur, len(rows), WIN))
    print("  起点   音色    F0     平坦度  发声%  150-300  1k-2k   4k-8k")
    if os.environ.get("DRIFT_FULL"):
      for r in rows:
        print("  %5.0fs %.4f  %6.1f  %.4f  %.3f  %.4f  %.4f  %.4f"
              % (r["t"], r["sim"], r["f0"], r["flat"], r["voiced"],
                 r["b150_300"], r["b1000_2000"], r["b4000_8000"]))
    for k, lab in (("sim", "音色"), ("f0", "F0"), ("flat", "平坦度"),
                   ("voiced", "发声占比"), ("b150_300", "150-300"), ("b4000_8000", "4k-8k")):
        v = np.array([r[k] for r in rows], dtype=float)
        h = len(v) // 3
        print("  %-6s 前1/3=%.4f 中1/3=%.4f 后1/3=%.4f  末-首=%+.4f  极差=%.4f"
              % (lab, v[:h].mean(), v[h:2 * h].mean(), v[2 * h:].mean(),
                 v[-1] - v[0], v.max() - v.min()))
    return rows


def main():
    args = sys.argv[1:]
    if args:
        for i, path in enumerate(args):
            rows, dur = run("f%d" % i, path)
            show(os.path.basename(path), rows, dur)
    else:
        ro, do = run("ours", OURS)
        re_, de = run("el", EL)
        show("我们的 h5 线上成品", ro, do)
        show("EL 案例（对照）", re_, de)
    print("\nDRIFT_DONE")


if __name__ == "__main__":
    main()
