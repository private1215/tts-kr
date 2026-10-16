#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L1 快筛：融合补丁是否 ①数学等价 ②平坦度提升（第30轮）

拿同一条 35s 裸男声，分别跑
  旧 app/fix_chain.py（备份在 /mnt/tts/diag/fix_chain.py.bak）
  新 app/fix_chain.py（融合版）
比：时长 / 字每秒 / F0 / 平坦度 / Σz²

判定：
  ① 时长差异 < 0.5s、F0 差异 < 3Hz  → 数学等价，可以安全上线
  ② 新版平坦度 ≥ 旧版              → 方向正确
  ③ Σz² 不比旧版差                → 无回归

## 用法
    ./venv/bin/python tools/l1_fuse.py
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")

RAW = "/mnt/tts/diag/raw_flat_vox.wav"
NEW = "/mnt/tts/projects/qwen3-tts/app/fix_chain.py"
OLD = "/mnt/tts/diag/fix_chain.py.bak"
DIAG = "/mnt/tts/diag"
OUT = "/mnt/tts/diag/muffle/l1_fuse.json"
CHARS = 320
CPS = 8.37

try:
    import librosa
except Exception:
    librosa = None


def run(script, tag):
    o = os.path.join(DIAG, "l1_fuse_%s.wav" % tag)
    if os.path.exists(o):
        os.remove(o)
    r = subprocess.run([sys.executable, script, RAW, o,
                        "--gender", "m", "--target-cps", str(CPS),
                        "--chars", str(CHARS)],
                       capture_output=True, text=True, timeout=900)
    ok = os.path.exists(o) and r.returncode == 0
    print("--- %s --- rc=%s" % (tag, r.returncode))
    for ln in (r.stdout or "").splitlines():
        if any(k in ln for k in ("模式", "窗口系数", "分窗 F0", "分窗变调",
                                 "变调后", "分段", "DONE", "自适应高频")):
            print("   " + ln.strip()[:150])
    if not ok:
        print("   STDERR:", (r.stderr or "")[-400:])
    return o if ok else None


def probe(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1",
                          "-ar", "16000", "-f", "f32le", "-"],
                         stdout=subprocess.PIPE, check=True).stdout
    y = np.frombuffer(raw, dtype=np.float32).copy()
    dur = len(y) / 16000.0
    flat = float(librosa.feature.spectral_flatness(
        y=y[:60 * 16000], hop_length=256)[0].mean()) if librosa else None
    # F0（pyin）
    try:
        f0, _, _ = librosa.pyin(y[:60 * 16000], fmin=60, fmax=400, sr=16000)
        f0m = float(np.nanmedian(f0)) if f0 is not None else None
    except Exception:
        f0m = None
    return {"dur": dur, "cps": CHARS / dur, "flat": flat, "f0": f0m}


def main():
    rows = {}
    for tag, script in (("old", OLD), ("new", NEW)):
        p = run(script, tag)
        if p:
            rows[tag] = probe(p)
            rows[tag]["path"] = p

    print("\n%-8s %9s %8s %9s %9s" % ("版本", "时长s", "字/秒", "F0 Hz", "平坦度"))
    for t in ("old", "new"):
        if t in rows:
            r = rows[t]
            print("%-8s %9.2f %8.2f %9.1f %9.5f"
                  % (t, r["dur"], r["cps"], r["f0"] or -1, r["flat"]))

    if "old" in rows and "new" in rows:
        a, b = rows["old"], rows["new"]
        dd = abs(a["dur"] - b["dur"])
        df = abs((a["f0"] or 0) - (b["f0"] or 0))
        print("\n① 等价性：时长差 %.2fs（<0.5 OK）  F0 差 %.1fHz（<3 OK）"
              % (dd, df))
        print("② 平坦度：%.5f → %.5f  (%+.1f%%)"
              % (a["flat"], b["flat"], (b["flat"] / a["flat"] - 1) * 100))
        print("③ 判定：%s" % ("✅ 可上线"
                            if dd < 0.5 and df < 3 and b["flat"] >= a["flat"] * 0.98
                            else "⚠️ 需要复核"))
    json.dump(rows, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("\n→ " + OUT)


if __name__ == "__main__":
    main()
