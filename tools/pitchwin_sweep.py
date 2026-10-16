#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第31轮：变调窗长扫描（60 / 30 / 15 秒），选 F0 最稳的（自动化，不需人工）

## 背景
融合补丁后跨窗一致性大部分改善，但 **F0 跨窗极差反而变差**：
    r30fix 15.9% → r31 21.2%（EL 15.1%）
逐窗看是两个离群窗（174.7Hz、217.6Hz）。根因：变调按 60s 校正，
漂移发生在**窗内**（第6/7窗 174.7 与 203.0 同属一个 60s 校正窗）。

## 做法
同一条裸输出（`raw_long_male.wav`，447s），跑 `--pitch-win 60 / 30 / 15`，
比：时长 / 字每秒 / **F0 跨窗极差** / 平坦度(全篇) / Σz²(probe_g)
选 F0 最稳且 Σz² 不退化的一档。

## 用法
    ./venv/bin/python tools/pitchwin_sweep.py
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
DIAG = "/mnt/tts/diag"
RAW = os.path.join(DIAG, "raw_long_male.wav")
FX = "/mnt/tts/projects/qwen3-tts/app/fix_chain.py"
OUT = "/mnt/tts/diag/muffle/pitchwin_sweep.json"
CHARS = 4143
CPS = 8.37
SR = 16000

try:
    import librosa
except Exception:
    librosa = None


def load16(p):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-ac", "1",
                          "-ar", str(SR), "-f", "f32le", "-"],
                         stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def dur(p):
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", p], stdout=subprocess.PIPE, text=True).stdout.strip())


def f0_series(y, win=30.0):
    n = int(win * SR)
    skip = int(len(y) * 0.05)
    out, s = [], skip
    while s + n <= len(y) - skip:
        f0, _, _ = librosa.pyin(y[s:s + n], fmin=60, fmax=400, sr=SR)
        out.append(float(np.nanmedian(f0)) if f0 is not None else None)
        s += n
    v = np.array([x for x in out if x and np.isfinite(x)])
    return v


def flat_full(y):
    return float(librosa.feature.spectral_flatness(
        y=y, hop_length=256)[0].mean())


def probe_g(p):
    r = subprocess.run([sys.executable, "/mnt/tts/diag/probe_g.py", "male", p],
                       capture_output=True, text=True,
                       cwd="/mnt/tts/projects/qwen3-tts")
    sig = None
    for ln in (r.stdout or "").splitlines():
        if "Σz²=" in ln:
            try:
                sig = float(ln.split("Σz²=")[1].split()[0])
            except Exception:
                pass
    return sig


def run(pw):
    o = os.path.join(DIAG, "pw_%d.wav" % int(pw))
    if os.path.exists(o):
        os.remove(o)
    subprocess.run([sys.executable, FX, RAW, o, "--gender", "m",
                    "--target-cps", str(CPS), "--chars", str(CHARS),
                    "--pitch-win", str(pw)],
                   capture_output=True, text=True, timeout=2400)
    if not os.path.exists(o):
        return None
    y = load16(o)
    f0 = f0_series(y)
    d = dur(o)
    return {"pw": pw, "path": o, "dur": d, "cps": CHARS / d,
            "f0_med": float(np.median(f0)),
            "f0_range_hz": float(f0.max() - f0.min()),
            "f0_range_pct": float((f0.max() - f0.min()) / np.median(f0) * 100),
            "flat_full": flat_full(y),
            "sigz2": probe_g(o), "series": [round(x, 1) for x in f0]}


def main():
    rows = []
    for pw in (60.0, 30.0, 15.0):
        r = run(pw)
        if r is None:
            print("❌ pitch-win=%.0f 失败" % pw)
            continue
        rows.append(r)
        print("\n--- pitch-win = %.0fs ---" % pw)
        print("  时长 %.2fs  字/秒 %.2f" % (r["dur"], r["cps"]))
        print("  F0 中位 %.1f Hz  极差 %.1f Hz (%.1f%%)"
              % (r["f0_med"], r["f0_range_hz"], r["f0_range_pct"]))
        print("  平坦度(全篇) %.5f   Σz² %s"
              % (r["flat_full"], r["sigz2"]))
        print("  逐窗 F0:", r["series"])

    print("\n\n=== 汇总 ===")
    print("%-12s %9s %8s %10s %10s %10s %8s"
          % ("pitch-win", "时长s", "字/秒", "F0极差Hz", "F0极差%", "平坦度", "Σz²"))
    for r in rows:
        print("%-12s %9.2f %8.2f %10.1f %10.1f %10.5f %8s"
              % ("%.0fs" % r["pw"], r["dur"], r["cps"], r["f0_range_hz"],
                 r["f0_range_pct"], r["flat_full"], r["sigz2"]))
    print("\nEL 参照：F0 极差 29.8 Hz (15.1%)，平坦度(全篇) 0.04329，Σz² 中位 59.6")

    best = min(rows, key=lambda r: r["f0_range_pct"]) if rows else None
    if best:
        print("\nF0 最稳：pitch-win=%.0fs（极差 %.1f Hz / %.1f%%）"
              % (best["pw"], best["f0_range_hz"], best["f0_range_pct"]))
    json.dump(rows, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("→ " + OUT)


if __name__ == "__main__":
    main()
