#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第31轮：变调窗长**跨文案交叉验证**（防样本级异常）

铁律：至少 2 篇不同文案。上一篇（6cdd）测出 15s 窗 F0 极差 10.2%
优于 60s 窗的 12.5%，但单一样本不能定案 —— 换一篇最长的文案再测一遍。

只比 60s（原）与 15s（候选）两档，省 GPU 时间。
不比 Σz²（跨文案不可比，各篇频谱基准不同），只比**相对改善**。

## 用法
    ./venv/bin/python tools/pitchwin_xcheck.py
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
DIAG = "/mnt/tts/diag"
FX = "/mnt/tts/projects/qwen3-tts/app/fix_chain.py"
OUT = "/mnt/tts/diag/muffle/pitchwin_xcheck.json"
SR = 16000

CASES = [
    ("6cdd", "/mnt/tts/data/cases/6cdd25bc75f2/full.txt",
     "/mnt/tts/diag/raw_long_male.wav"),
    ("d172", "/mnt/tts/data/cases/d172b8c4efb3/full.txt",
     "/mnt/tts/diag/raw_long_male_d172.wav"),
]

try:
    import librosa
except Exception:
    librosa = None


def load16(p):
    return np.frombuffer(subprocess.run(
        ["ffmpeg", "-v", "error", "-i", p, "-ac", "1", "-ar", str(SR),
         "-f", "f32le", "-"], stdout=subprocess.PIPE,
        check=True).stdout, dtype=np.float32).copy()


def dur(p):
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", p], stdout=subprocess.PIPE, text=True).stdout.strip())


def f0_stats(y, win=30.0):
    n = int(win * SR)
    skip = int(len(y) * 0.05)
    out, s = [], skip
    while s + n <= len(y) - skip:
        f0, _, _ = librosa.pyin(y[s:s + n], fmin=60, fmax=400, sr=SR)
        out.append(float(np.nanmedian(f0)) if f0 is not None else None)
        s += n
    v = np.array([x for x in out if x and np.isfinite(x)])
    return (float(np.median(v)), float(v.max() - v.min()),
            float((v.max() - v.min()) / np.median(v) * 100))


def run(raw, chars, pw, tag):
    o = os.path.join(DIAG, "x_%s_pw%d.wav" % (tag, int(pw)))
    if os.path.exists(o):
        os.remove(o)
    r = subprocess.run([sys.executable, FX, raw, o, "--gender", "m",
                        "--target-cps", "8.37", "--chars", str(chars),
                        "--pitch-win", str(pw)],
                       capture_output=True, text=True, timeout=2400)
    if not os.path.exists(o):
        print("❌ %s pw=%.0f 失败: %s" % (tag, pw, (r.stderr or "")[-200:]))
        return None
    med, rng, pct = f0_stats(load16(o))
    return {"case": tag, "pw": pw, "dur": dur(o),
            "f0_med": med, "f0_range_hz": rng, "f0_range_pct": pct}


def main():
    rows = []
    for tag, txt_p, raw in CASES:
        if not os.path.exists(raw):
            print("缺裸输出 %s（先跑 tools/gen_raw_long.py）" % raw)
            continue
        chars = len(open(txt_p, encoding="utf-8").read().strip())
        for pw in (60.0, 15.0):
            r = run(raw, chars, pw, tag)
            if r:
                rows.append(r)
                print("%-6s pw=%-4.0f  F0 中位 %6.1f  极差 %5.1f Hz (%5.1f%%)"
                      % (tag, pw, r["f0_med"], r["f0_range_hz"],
                         r["f0_range_pct"]))

    print("\n=== 跨文案汇总（EL 参照 极差 29.8 Hz / 15.1%）===")
    print("%-8s %10s %10s %10s" % ("文案", "pw=60", "pw=15", "改善"))
    for tag, _, _ in CASES:
        a = [r for r in rows if r["case"] == tag and r["pw"] == 60.0]
        b = [r for r in rows if r["case"] == tag and r["pw"] == 15.0]
        if a and b:
            d = b[0]["f0_range_pct"] - a[0]["f0_range_pct"]
            print("%-8s %9.1f%% %9.1f%% %9s"
                  % (tag, a[0]["f0_range_pct"], b[0]["f0_range_pct"],
                     "%+.1f 个百分点" % d))
    json.dump(rows, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("→ " + OUT)


if __name__ == "__main__":
    main()
