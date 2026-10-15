#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""停顿底噪下压（downward expander）候选（2026-10-15 第30轮）

## 动机（实测，非推理）
10ms 帧统计「静音帧」（<−45dB）电平：

| | 中位 | p5 | <−90dB 占比 | 数字零帧 |
|---|---|---|---|---|
| 我们（现役） | **−51.80** | −75.25 | 1.17% | 54 |
| EL 标杆 | **−56.60** | −89.04 | 4.41% | 1 |
| EL 过 mp3 后 | −56.72 | −89.36 | 4.41% | 0 |

🔴 **EL 过一遍 mp3 后中位几乎不变（−56.60→−56.72）→ 这 4.9dB 不是 mp3 假象，
   是我们成品真实的底噪差距。** EL 的停顿「真的静得下去」（4.4% 深到 −90dB 以下），
   我们几乎一直在 −52dB 挂着一层底噪 → 很可能就是员工说的「电子合成感 / 整体不像真人」。

## 做法
对 <−45dB 的部分做**向下扩展** ratio>1：
    gain_db = (level_db − thr) * (ratio − 1)
- 用**对称窗**（非因果包络跟随）→ 没有 attack/release，不会抽气/抽水
- 增益曲线再做 ~30ms 平滑 → 过渡柔和
- 只作用于 −45dB 以下，不动任何可听内容

## 目标映射（thr=−45）
| ratio | 中位 −51.8 → | p5 −75.3 → |
|---|---|---|
| 1.5 | −55.3 | −90.4 |
| 1.6 | −56.0 | −93.7 |
| 1.8 | −57.2 | −99.6 |
EL 目标：中位 −56.7 / p5 −89.4
"""
import json
import os
import subprocess
import sys

import numpy as np

SR = 24000
DIAG = "/mnt/tts/diag/muffle"
OUT_JSON = os.path.join(DIAG, "floor_tune.json")

FRAME_MS = 10.0
QUIET_FRAME_DB = -45.0
HARD_DB = -60.0


def load(path, sr=SR):
    cmd = ["ffmpeg", "-v", "error", "-i", path,
           "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def expand(x, sr=SR, thr_db=-45.0, ratio=1.5, win_ms=10.0, smooth_ms=30.0):
    """向下扩展：把 <thr 的部分按 ratio 压得更低。不改长度。"""
    n = max(1, int(sr * win_ms / 1000.0))
    nfr = len(x) // n
    if nfr < 2:
        return x
    fr = x[:nfr * n].reshape(nfr, n)
    rms = np.sqrt((fr ** 2).mean(axis=1) + 1e-30)
    lvl = 20 * np.log10(rms + 1e-30)
    g = np.where(lvl > thr_db, 0.0, (lvl - thr_db) * (ratio - 1.0))

    # 对称平滑增益曲线（Hann），避免抽气
    k = max(1, int(round(smooth_ms / win_ms)))
    if k > 1:
        if k % 2 == 0:
            k += 1
        w = np.hanning(k)
        w /= w.sum()
        pad = k // 2
        gp = np.concatenate([np.full(pad, g[0]), g, np.full(pad, g[-1])])
        g = np.convolve(gp, w, mode="valid")

    # 线性插值到样本级
    xs = (np.arange(nfr) + 0.5) * n
    xq = np.arange(nfr * n)
    gs = np.interp(xq, xs, g).astype(np.float32)
    out = x.copy()
    out[:nfr * n] = out[:nfr * n] * (10 ** (gs / 20.0))
    if len(x) > nfr * n:          # 尾巴沿用最后一帧增益
        out[nfr * n:] = out[nfr * n:] * (10 ** (g[-1] / 20.0))
    return out


def measure(x, sr=SR, name=""):
    n = int(sr * FRAME_MS / 1000.0)
    nfr = len(x) // n
    fr = x[:nfr * n].reshape(nfr, n)
    rms = np.sqrt((fr ** 2).mean(axis=1) + 1e-30)
    db = 20 * np.log10(rms + 1e-30)
    quiet = db < QUIET_FRAME_DB
    qdb = db[quiet]
    d = np.diff(db)
    return {
        "name": name,
        "dur_s": round(len(x) / float(sr), 2),
        "med_db": round(float(np.median(qdb)), 2),
        "sd_db": round(float(qdb.std()), 2),
        "p5_db": round(float(np.percentile(qdb, 5)), 2),
        "lt90_pct": round(float(100.0 * (qdb < -90).mean()), 2),
        "hard_cuts": int((d < HARD_DB).sum()),
        "steepest5_db": round(float(np.sort(d)[:5].mean()), 2),
        "zero_frames": int((db < -200).sum()),
    }


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import dc_tune as T
    base = T.load(os.path.join(DIAG, "base24.wav"))
    rows = [
        measure(base, name="现役(未下压)"),
        measure(T.load(os.path.join(DIAG, "el24.wav")), name="EL标杆"),
        measure(T.load(os.path.join(DIAG, "el_mp3.mp3")), name="EL过mp3"),
    ]
    for ratio in (1.4, 1.5, 1.6, 1.8):
        y = expand(base, ratio=ratio)
        y = T.declick(y, fill_db=-89.0, quiet_db=-100.0, mode="floor")
        rows.append(measure(y, name="expand r%.1f + declick" % ratio))

    h = ("%-24s %7s %7s %8s %8s %6s %9s %6s"
         % ("name", "med", "sd", "p5", "<90%", "hard", "steep5", "zerofr"))
    print(h)
    print("-" * len(h))
    for r in rows:
        print("%-24s %7.2f %7.2f %8.2f %8.2f %6d %9.2f %6d"
              % (r["name"], r["med_db"], r["sd_db"], r["p5_db"],
                 r["lt90_pct"], r["hard_cuts"], r["steepest5_db"],
                 r["zero_frames"]))
    with open(OUT_JSON, "w") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    print("\n→ " + OUT_JSON)


if __name__ == "__main__":
    main()
