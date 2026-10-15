#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""declick 精调对照实验（2026-10-15 第29轮后）

## 要解决什么
第 29 轮的 declick（整段替换）已经把「硬切」从 7 降到 0，
但把停顿帧电平的 σ 从 14.20dB 压到 7.99dB（EL 是 13.19dB）。
→ 因为它**整段替换**了本来就有自然起伏的停顿。

## 新思路：逐样本「幅度下限」floor 模式
    out = where(|噪声| > |原信号|, 噪声, 原信号)
- 原本有起伏的地方（电平高于噪声）→ **原样不动**
- 真正掉到数字零的地方 → 被噪声顶上来
再配合**更严的门限**（只认 −85/−95/−105dB 这种真·数字零），
把 σ 保住，同时硬切清零。

## 用法
    python3 tools/dc_tune.py
产出 JSON + 打印对照表。
"""
import json
import os
import subprocess
import sys

import numpy as np

SR = 24000
DIAG = "/mnt/tts/diag/muffle"
BASE_MP3 = "/mnt/tts/output/longprobe-omni-1block/longprobe-omni-1block-1.mp3"
EL_WAV = "/mnt/tts/data/cases/6cdd25bc75f2/merged.wav"
OUT_JSON = os.path.join(DIAG, "dc_tune.json")

FRAME_MS = 10.0        # 帧长（与第29轮口径一致）
QUIET_FRAME_DB = -45.0  # 低于此电平的帧算「停顿帧」
HARD_DB = -60.0        # 帧间落差超过 60dB 算「硬切」


def load(path, sr=SR):
    """统一解到 24k 单声道 float32。"""
    cmd = ["ffmpeg", "-v", "error", "-i", path,
           "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


# ---------------- 噪声生成（与 app/declick.py 完全一致） ----------------
def _pink(n, rng):
    m = 1 << int(np.ceil(np.log2(max(n, 2))))
    w = np.fft.rfft(rng.standard_normal(m))
    f = np.fft.rfftfreq(m, 1.0)
    amp = np.ones_like(f)
    lo = f < 100.0
    amp[lo] = 0.5 * (1.0 - np.cos(np.pi * f[lo] / 100.0))
    hi = f > 3000.0
    amp[hi] = 0.35 * (1.0 - 0.5 * (1.0 + np.cos(
        np.pi * np.clip((f[hi] - 3000.0) / 3000.0, 0, 1))))
    y = np.fft.irfft(w * amp, n=m)
    return y[:n]


def _room_tone(n, sr, rng, fill_db):
    base = _pink(n, rng)
    base /= (np.abs(base).max() + 1e-20)
    rms0 = np.sqrt((base ** 2).mean()) + 1e-20
    return (base * (10 ** (fill_db / 20.0) / rms0)).astype(np.float32)


def _runs(quiet):
    """返回连续 True 段的 (start, end) 列表。"""
    q = quiet.astype(np.int8)
    edges = np.diff(q)
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if q[0]:
        starts.insert(0, 0)
    if q[-1]:
        ends.append(len(q))
    return starts, ends


def declick(x, sr=SR, fill_db=-66.0, quiet_db=-72.0, mode="floor"):
    """mode=replace → 第29轮现役（整段替换）
       mode=floor   → 新方案（逐样本幅度下限，保住原有起伏）"""
    y = np.asarray(x, dtype=np.float32)
    thr = 10 ** (quiet_db / 20.0)
    quiet = np.abs(y) < thr
    if not quiet.any():
        return y
    rng = np.random.default_rng(1234)
    out = y.copy()
    min_len = int(sr * 0.002)
    starts, ends = _runs(quiet)
    ramp = max(1, int(sr * 0.001))
    for s, e in zip(starts, ends):
        a, b = int(s), int(e)
        if b - a < min_len:
            continue
        tone = _room_tone(b - a, sr, rng, fill_db)
        rl = min(ramp, (b - a) // 2)
        env_r = np.ones(b - a, dtype=np.float32)
        if rl > 0:
            env_r[:rl] = np.sin(np.pi * np.linspace(0, 1, rl, dtype=np.float32))
            env_r[-rl:] = np.sin(np.pi * np.linspace(1, 0, rl, dtype=np.float32))
        tone = tone * env_r
        if mode == "replace":
            out[a:b] = tone
        else:  # floor
            seg = out[a:b]
            out[a:b] = np.where(np.abs(tone) > np.abs(seg), tone, seg)
    return out


# ---------------- 测量 ----------------
def measure(x, sr=SR, name=""):
    n = int(sr * FRAME_MS / 1000.0)
    nfr = len(x) // n
    fr = x[:nfr * n].reshape(nfr, n)
    rms = np.sqrt((fr ** 2).mean(axis=1) + 1e-30)
    db = 20 * np.log10(rms + 1e-30)
    quiet = db < QUIET_FRAME_DB
    qdb = db[quiet]
    # 硬切：相邻帧落差 < -60dB（只统计落在停顿里/落入停顿的那一下）
    d = np.diff(db)
    hard = int((d < HARD_DB).sum())
    steep = float(np.sort(d)[:5].mean()) if len(d) >= 5 else float(d.min())
    return {
        "name": name,
        "dur_s": round(len(x) / float(sr), 2),
        "n_frame": int(nfr),
        "n_quiet_frame": int(quiet.sum()),
        "med_db": round(float(np.median(qdb)), 2),
        "sd_db": round(float(qdb.std()), 2),
        "p5_db": round(float(np.percentile(qdb, 5)), 2),
        "p95_db": round(float(np.percentile(qdb, 95)), 2),
        "span_db": round(float(np.percentile(qdb, 95) - np.percentile(qdb, 5)), 2),
        "hard_cuts": hard,
        "steepest5_db": round(steep, 2),
        "zero_pct": round(float((np.abs(x) < 1e-9).mean() * 100), 4),
    }


def main():
    base_path = os.path.join(DIAG, "base24.wav")
    el_path = os.path.join(DIAG, "el24.wav")
    if not os.path.exists(base_path):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", BASE_MP3,
                        "-ac", "1", "-ar", str(SR), base_path], check=True)
    if not os.path.exists(el_path):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", EL_WAV,
                        "-ac", "1", "-ar", str(SR), el_path], check=True)

    base = load(base_path)
    el = load(el_path)

    rows = [measure(base, name="处理前(omni-1block)")]
    rows.append(measure(el, name="EL标杆(6cdd)"))

    # 现役（第29轮）
    rows.append(measure(declick(base, fill_db=-66.0, quiet_db=-72.0,
                                mode="replace"),
                        name="现役 replace q-72 f-66"))

    # 新方案网格
    for q in (-72.0, -85.0, -95.0, -105.0):
        for f in (-60.0, -66.0, -72.0):
            y = declick(base, fill_db=f, quiet_db=q, mode="floor")
            rows.append(measure(y, name="floor q%g f%g" % (q, f)))

    hdr = ("%-24s %6s %7s %7s %7s %7s %7s %6s %8s"
           % ("name", "dur", "med", "sd", "p5", "p95", "span", "hard", "steep5"))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print("%-24s %6.1f %7.2f %7.2f %7.2f %7.2f %7.2f %6d %8.2f"
              % (r["name"], r["dur_s"], r["med_db"], r["sd_db"],
                 r["p5_db"], r["p95_db"], r["span_db"],
                 r["hard_cuts"], r["steepest5_db"]))

    with open(OUT_JSON, "w") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    print("\n→ " + OUT_JSON)


if __name__ == "__main__":
    sys.exit(main())
