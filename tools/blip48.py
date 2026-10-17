# -*- coding: utf-8 -*-
"""blip48.py —— 找「孤立的短发声段」（员工说的「嘎哈」）。

## 与 gapburst48.py 的区别（🔴 关键）
gapburst48 先做 VAD 切出「停顿」，再在停顿里找**弱**突起。
但实测 xf5272-prod 那个音峰值 -15.3 dB，**比全篇均值高 16 dB** —— 它能量和
正常说话一样，**会被 VAD 判成发声**，根本进不了"停顿"，所以 gapburst48 检 0 处。

→ 本脚本反过来：先找**发声段**，再看哪些发声段是「孤立」的：
   时长 0.05~0.5s，且前后各有 ≥0.10s 的静音。
孤立 = 它不属于任何一个词，是凭空冒出来的。

## 判据（全部以全篇 mean 为基准，对 EL / 我们完全同口径）
- 发声帧：db > mean_db + VOICE_UP
- 静音帧：db < mean_db - SIL_DOWN
- 发声段：连续发声帧，间隙 <0.05s 合并
- blip：段时长 MIN~MAX，且前后静音各 ≥ GAP

用法：python3 blip48.py <音频...> [--json out.json]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 0.005
VOICE_UP = 3.0        # 高于全篇均值多少 dB 算发声
SIL_DOWN = 10.0       # 低于全篇均值多少 dB 算静音
MERGE = 0.05          # 发声段合并间隙
MIN_LEN, MAX_LEN = 0.05, 0.50
GAP = 0.10            # blip 前后至少这么长的静音


def load(p):
    if not p.endswith(".wav"):
        t = tempfile.mktemp(suffix=".wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                        "-ar", str(SR), "-ac", "1", t], check=True)
        p = t
    x, sr = sf.read(p, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def analyze(path):
    x, sr = load(path)
    dur = len(x) / sr
    n = int(sr * FRAME)
    m = (len(x) // n) * n
    fr = x[:m].reshape(-1, n)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-12))
    mean_db = float(db.mean())

    voiced = db > (mean_db + VOICE_UP)
    # 合并间隙
    mg = int(MERGE / FRAME)
    vv = voiced.copy()
    i = 0
    while i < len(vv):
        if not vv[i]:
            j = i
            while j + 1 < len(vv) and not vv[j + 1]:
                j += 1
            if (j - i + 1) <= mg:
                vv[i:j + 1] = True
            i = j + 1
        else:
            i += 1

    # 连续发声段
    segs, i = [], 0
    while i < len(vv):
        if vv[i]:
            j = i
            while j + 1 < len(vv) and vv[j + 1]:
                j += 1
            segs.append((i, j))
            i = j + 1
        else:
            i += 1

    sil = db < (mean_db - SIL_DOWN)
    def sil_run(a, b):
        """从 a 往前 / 从 b 往后 连续静音的时长（秒）"""
        k = a
        while k - 1 >= 0 and sil[k - 1]:
            k -= 1
        pre = (a - k) * FRAME
        k = b
        while k + 1 < len(sil) and sil[k + 1]:
            k += 1
        post = (k - b) * FRAME
        return pre, post

    blips = []
    for a, b in segs:
        d = (b - a + 1) * FRAME
        if not (MIN_LEN <= d <= MAX_LEN):
            continue
        pre, post = sil_run(a, b)
        if pre < GAP or post < GAP:
            continue
        peak = float(db[a:b + 1].max())
        blips.append({"t0": round(a * FRAME, 2), "t1": round((b + 1) * FRAME, 2),
                      "dur": round(d, 2),
                      "peak_db": round(peak, 1),
                      "rel_mean_db": round(peak - mean_db, 1),
                      "pre_sil": round(pre, 2), "post_sil": round(post, 2)})

    return {"file": os.path.basename(path), "dur": round(dur, 1),
            "mean_db": round(mean_db, 1),
            "n_seg": len(segs), "n_blip": len(blips),
            "per100s": round(len(blips) / dur * 100, 2) if dur > 0 else 0,
            "blips": blips[:40]}


def main():
    global GAP, SIL_DOWN, VOICE_UP, MIN_LEN, MAX_LEN
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--json", default="")
    ap.add_argument("--gap", type=float, default=GAP)
    ap.add_argument("--sil", type=float, default=SIL_DOWN)
    ap.add_argument("--up", type=float, default=VOICE_UP)
    ap.add_argument("--minlen", type=float, default=MIN_LEN)
    ap.add_argument("--maxlen", type=float, default=MAX_LEN)
    a = ap.parse_args()
    GAP, SIL_DOWN, VOICE_UP = a.gap, a.sil, a.up
    MIN_LEN, MAX_LEN = a.minlen, a.maxlen
    print("口径: 发声>mean+%.0f dB  静音<mean-%.0f dB  前后静音>=%.2fs  时长 %.2f~%.2fs"
          % (VOICE_UP, SIL_DOWN, GAP, MIN_LEN, MAX_LEN))
    rows = []
    for p in a.wavs:
        r = analyze(p)
        rows.append(r)
        print("%-36s %6.1fs  发声段%4d  孤立短音%3d  密度 %5.2f 处/100s"
              % (r["file"], r["dur"], r["n_seg"], r["n_blip"], r["per100s"]))
        for b in r["blips"][:6]:
            print("      %.2f~%.2fs  %.2fs  峰值 %+.1f dB(相对均值)  前静音%.2f 后静音%.2f"
                  % (b["t0"], b["t1"], b["dur"], b["rel_mean_db"],
                     b["pre_sil"], b["post_sil"]))
    if rows:
        d = [r["per100s"] for r in rows]
        print("\n合计 %d 条：密度 中位 %.2f  均值 %.2f 处/100s"
              % (len(rows), float(np.median(d)), float(np.mean(d))))
        tot = sum(r["n_blip"] for r in rows); td = sum(r["dur"] for r in rows)
        print("汇总：%d 处 / %.1f 分钟 = %.2f 处/100s" % (tot, td / 60, tot / td * 100))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        print("saved", a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
