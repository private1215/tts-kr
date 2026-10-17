# -*- coding: utf-8 -*-
"""reburst48.py —— 用**归一化包络形态**找「说完收干净了又冒出来的短音」。

## 为什么不用绝对 dB 阈值（踩了两次）
xf5272-prod 22.82~23.03s 那个音，前面 22.79s 是 -33.0 dB：
- 按「>mean+3 算发声」它不是发声；
- 按「<mean-6 算静音」它也不是静音；
卡在中间地带，任何绝对阈值都会漏。

而且实测**我们和 EL 的能量分布几乎完全一致**
（发声帧中位 -19.4 vs -19.0~-19.3；深静音 p5 -53~-54 vs -53~-56）
→ 差异根本不在响度，在**时序结构**。

## 本脚本的做法：归一化到自身动态范围，只看形态
1. 5ms 帧 dB 包络
2. `lo = p5`、`hi = p95` → `norm = (db-lo)/(hi-lo)`，clamp [0,1]
   （用自身分位数，天然抵消两条素材的整体响度差）
3. 20ms 中值平滑去毛刺
4. **深谷** norm < 0.25 → **回升** norm > 0.75 → **再落回** norm < 0.25
5. 回升段时长 0.05~0.60s 才算「孤立短音」
   （>0.60s 是正常的下一句话，不算）

用法：python3 reburst48.py <音频...> [--json out.json]
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
LOW, HIGH = 0.25, 0.75
MIN_LEN, MAX_LEN = 0.05, 0.60
SMOOTH = 0.02


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


def medfilt(v, win):
    """简易中值滤波（避免依赖 scipy）"""
    k = max(1, int(win / FRAME))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    vp = np.pad(v, pad, mode="edge")
    out = np.empty_like(v)
    for i in range(len(v)):
        out[i] = np.median(vp[i:i + k])
    return out


def analyze(path):
    x, sr = load(path)
    dur = len(x) / sr
    n = int(sr * FRAME)
    m = (len(x) // n) * n
    rms = np.sqrt((x[:m].reshape(-1, n) ** 2).mean(axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-12))
    lo, hi = float(np.percentile(db, 5)), float(np.percentile(db, 95))
    norm = np.clip((db - lo) / max(hi - lo, 1e-6), 0, 1)
    nv = medfilt(norm, SMOOTH)

    # 找「深谷 → 回升 → 再落回」的形态
    deep = nv < LOW
    high = nv > HIGH
    # 每个"高台"：连续 high
    peaks, i = [], 0
    while i < len(high):
        if high[i]:
            j = i
            while j + 1 < len(high) and high[j + 1]:
                j += 1
            peaks.append((i, j))
            i = j + 1
        else:
            i += 1

    blips = []
    for a, b in peaks:
        # 前面必须有深谷
        k = a
        while k - 1 >= 0 and not deep[k - 1]:
            k -= 1
        if k - 1 < 0:
            continue
        pre_deep_end = k - 1               # 深谷最后一帧
        # 深谷起点
        k2 = pre_deep_end
        while k2 - 1 >= 0 and deep[k2 - 1]:
            k2 -= 1
        pre_deep_len = (pre_deep_end - k2 + 1) * FRAME
        if pre_deep_len < 0.02:
            continue
        # 后面必须落回深谷
        k3 = b
        while k3 + 1 < len(deep) and not deep[k3 + 1]:
            k3 += 1
        if k3 + 1 >= len(deep):
            continue
        post_deep_start = k3 + 1
        k4 = post_deep_start
        while k4 + 1 < len(deep) and deep[k4 + 1]:
            k4 += 1
        post_deep_len = (k4 - post_deep_start + 1) * FRAME
        if post_deep_len < 0.02:
            continue
        # 回升段时长：深谷结束 → 深谷开始
        d = (post_deep_start - pre_deep_end) * FRAME
        if not (MIN_LEN <= d <= MAX_LEN):
            continue
        peak_db = float(db[a:b + 1].max())
        blips.append({"t0": round(pre_deep_end * FRAME, 2),
                      "t1": round(post_deep_start * FRAME, 2),
                      "dur": round(d, 2),
                      "peak_db": round(peak_db, 1),
                      "rel_hi_db": round(peak_db - hi, 1),
                      "norm_peak": round(float(nv[a:b + 1].max()), 2),
                      "pre_deep": round(pre_deep_len, 2),
                      "post_deep": round(post_deep_len, 2)})

    return {"file": os.path.basename(path), "dur": round(dur, 1),
            "lo_db": round(lo, 1), "hi_db": round(hi, 1),
            "n_blip": len(blips),
            "per100s": round(len(blips) / dur * 100, 2) if dur > 0 else 0,
            "blips": blips[:40]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rows = []
    for p in a.wavs:
        r = analyze(p)
        rows.append(r)
        print("%-34s %6.1fs  孤立短音%3d  密度 %5.2f 处/100s"
              % (r["file"], r["dur"], r["n_blip"], r["per100s"]))
        for b in r["blips"][:5]:
            print("      %.2f~%.2fs %.2fs  峰值 %+.1f dB(相对p95)  "
                  "前谷%.2f 后谷%.2f" % (b["t0"], b["t1"], b["dur"],
                                        b["rel_hi_db"], b["pre_deep"], b["post_deep"]))
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
