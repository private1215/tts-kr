#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""「像机器人」客观诊断：对同一时间窗比较韵律自然度指标。

员工反馈「开始就能听出来是机器人」。本脚本量化「真人感」相关的客观维度，
对比 EL 标杆 与 我方各版本（同取开头 60s，避免长文尾部漂移干扰）。

指标（都是韵律自然度经典量，不是音色相似度）：
  f0_med / f0_std / f0_iqr   音高绝对水平与起伏幅度
  f0_jit                     F0 相邻帧相对差分 std —— 真人在持续动，TTS 常常过平
  f0_range_p5_p95            音高动态范围
  env_db_p95_p5              能量包络动态范围（真人大）
  dur_cv                     发声段时长变异系数（真人节奏不均）
  hf_ratio                   >4kHz 能量占比（过高=金属/齿音感）
  flat                       频谱平坦度（过低=机械，过高=沙沙）
  cps                        字/秒（需传 --chars）
"""
import argparse
import subprocess
import tempfile
import os

import numpy as np
import soundfile as sf
import librosa


def load16(path):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "16000", "-ac", "1", t], check=True)
    y, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    return y, 16000


def analyze(path, dur=60.0, chars=None):
    y, sr = load16(path)
    n = int(dur * sr)
    y = y[:n]
    out = {"path": path, "sec": round(len(y) / sr, 1)}

    # ---- F0 (pyin) ----
    f0, vo, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr,
                             frame_length=1024, hop_length=256)
    f0v = f0[(vo if vo is not None else np.zeros_like(f0, dtype=bool)) & np.isfinite(f0)]
    if len(f0v) > 50:
        out["f0_med"] = round(float(np.median(f0v)), 1)
        out["f0_std"] = round(float(np.std(f0v)), 1)
        out["f0_p5"] = round(float(np.percentile(f0v, 5)), 1)
        out["f0_p95"] = round(float(np.percentile(f0v, 95)), 1)
        out["f0_range"] = round(out["f0_p95"] - out["f0_p5"], 1)
        # jitter-like: 相邻有声帧相对差分
        d = np.diff(f0v) / np.maximum(f0v[:-1], 1e-6)
        d = d[np.abs(d) < 0.5]
        out["f0_jit"] = round(float(np.std(d)), 4)
    else:
        for k in ("f0_med", "f0_std", "f0_p5", "f0_p95", "f0_range", "f0_jit"):
            out[k] = None

    # ---- 能量包络 ----
    hop = 256
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
    db = 20 * np.log10(np.maximum(rms, 1e-8))
    out["env_db_p95_p5"] = round(float(np.percentile(db, 95) - np.percentile(db, 5)), 1)
    out["env_db_std"] = round(float(np.std(db)), 2)

    # ---- 发声段时长变异 ----
    thr = np.percentile(db, 20) + 6
    voiced = db > thr
    # 连续段长度
    ch = np.diff(np.concatenate(([0], voiced.view(np.int8), [0])))
    st = np.where(ch == 1)[0]
    en = np.where(ch == -1)[0]
    segs = (en - st) * hop / sr
    segs = segs[segs > 0.05]
    out["n_voiced_seg"] = int(len(segs))
    out["dur_cv"] = round(float(np.std(segs) / max(np.mean(segs), 1e-6)), 3) if len(segs) > 3 else None
    out["dur_mean"] = round(float(np.mean(segs)), 3) if len(segs) else None

    # ---- 频谱 ----
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    tot = S.sum()
    out["hf_ratio"] = round(float(S[freqs > 4000].sum() / max(tot, 1e-12)), 4)
    out["mid_ratio"] = round(float(S[(freqs > 300) & (freqs < 3400)].sum() / max(tot, 1e-12)), 4)
    flat = librosa.feature.spectral_flatness(y=y, hop_length=256)[0]
    out["flat_mean"] = round(float(np.mean(flat)), 5)
    sc = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=256)[0]
    out["centroid"] = round(float(np.mean(sc)), 0)

    if chars:
        out["cps"] = round(chars / (len(y) / sr), 2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--dur", type=float, default=60.0)
    ap.add_argument("--chars", type=float, default=None)
    a = ap.parse_args()
    rows = []
    for w in a.wavs:
        try:
            rows.append(analyze(w, a.dur, a.chars))
        except Exception as e:
            print("FAIL", w, repr(e)[:160])
    keys = ["path", "sec", "f0_med", "f0_std", "f0_range", "f0_jit",
            "env_db_p95_p5", "dur_cv", "dur_mean", "n_voiced_seg",
            "hf_ratio", "mid_ratio", "flat_mean", "centroid", "cps"]
    print("\t".join(keys))
    for r in rows:
        print("\t".join(str(r.get(k, "")) for k in keys))
