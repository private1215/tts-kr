#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修「断句有异声」：句末收尾收紧（软噪声门）

## 依据（attack_ab.py 实测，同文案对照真人原声）
    句末 release：我们 117.5ms  vs  真人 22.5ms   → 我们拖 5 倍
    句首 attack ：我们 157.5ms  vs  真人 290.0ms  → 我们更"蹦"（次要，且动它伤辅音）
    长停顿数量 ：我们 3.0 处/分 vs 真人 5.9 处/分 （次要，插静音风险高，本轮不动）

真人在句末是**干净收住**，我们是一条 ~120ms 的衰减尾巴。这条尾巴落在"断句"
的位置上，就是员工听到的「啊—— / 额——」异声。

## 做法：时变软噪声门（只对"很轻"的部分动手）
    HI = 发声均值 − 15 dB   高于它完全不动（真词语音都在这之上）
    LO = 静音地板 + 3 dB    低于它压到地板
    中间线性过渡，增益再做 30ms 平滑（防调制失真）
只削尾巴，不碰任何正常电平的语音 → 不会吃掉字。

## 验什么（L1）
    release 是否 117ms → ~30ms；Σz² 不许变差；音色不许掉。
"""

import os
import subprocess

import numpy as np
import soundfile as sf

HOP = 0.005
SMOOTH_MS = 30


def load16(path, tmp="/tmp/_tc"):
    os.makedirs(tmp, exist_ok=True)
    out = os.path.join(tmp, "x.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ac", "1", "-ar", "16000", out], check=True)
    return sf.read(out, dtype="float32")


def env_db(y, sr):
    hop = int(HOP * sr)
    n = len(y) // hop
    e = np.sqrt(np.array([(y[i * hop:(i + 1) * hop] ** 2).mean() for i in range(n)]))
    return 20 * np.log10(e + 1e-12)


def gain_curve(db, floor, voiced, below=15.0):
    hi = voiced - below
    lo = floor + 3.0
    if hi <= lo:
        hi = lo + 1.0
    t = np.clip((hi - db) / (hi - lo), 0.0, 1.0)     # 0=响(不动) 1=轻(压到底)
    return -(hi - lo) * t                            # dB


def smooth(x, ms):
    w = max(1, int(ms / 1000.0 / HOP))
    if w == 1:
        return x
    k = np.ones(w) / w
    pad = np.pad(x, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(pad, k, mode="valid")[:len(x)]


def process(src, dst, below=15.0):
    y, sr = load16(src, "/mnt/tts/diag/_tailcut")
    db = env_db(y, sr)
    floor = np.percentile(db, 10)
    voiced = db[db > floor + 12].mean()
    g = smooth(gain_curve(db, floor, voiced, below), SMOOTH_MS)
    # 插值到样本级
    gi = np.interp(np.arange(len(y)) / float(sr), np.arange(len(db)) * HOP, g)
    out = y * (10.0 ** (gi / 20.0)).astype(np.float32)
    peak = float(np.abs(out).max())
    if peak > 0.99:
        out = out * (0.99 / peak)
    sf.write(os.path.join("/mnt/tts/diag/_tailcut", "y.wav"), out, sr)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i",
                    os.path.join("/mnt/tts/diag/_tailcut", "y.wav"),
                    "-b:a", "192k", dst], check=True)
    return floor, voiced


def metrics(path, tag):
    """复用 attack_ab 的口径，只测 release / attack"""
    y, sr = load16(path, "/mnt/tts/diag/_tailcut")
    db = env_db(y, sr)
    floor = np.percentile(db, 10)
    voiced = db[db > floor + 12].mean()
    lo, hi = floor + 3, voiced - 3
    on = db > floor + 6
    d = np.diff(on.astype(np.int8))
    st = list(np.where(d == 1)[0] + 1)
    en = list(np.where(d == -1)[0] + 1)
    if on[0]:
        st.insert(0, 0)
    if on[-1]:
        en.append(len(on))
    m = min(len(st), len(en))
    seg = [(st[i], en[i]) for i in range(m)]
    rel, atk = [], []
    for k in range(len(seg) - 1):
        if (seg[k + 1][0] - seg[k][1]) * HOP < 0.25:
            continue
        pe, ge = seg[k][1], seg[k + 1][0]

        def cross(i0, i1, lvl, up):
            for i in range(i0, min(i1, len(db))):
                if (db[i] >= lvl) if up else (db[i] <= lvl):
                    return i
            return None
        i2 = cross(max(pe - 120, seg[k][0]), pe, hi, False)
        j2 = cross(i2 if i2 is not None else pe, min(pe + 130, len(db)), lo, False) if i2 is not None else None
        if i2 is not None and j2 is not None:
            rel.append((j2 - i2) * HOP * 1000)
        i = cross(ge, ge + 120, lo, True)
        j = cross(i if i is not None else ge, ge + 120, hi, True) if i is not None else None
        if i is not None and j is not None:
            atk.append((j - i) * HOP * 1000)
    print("  %-12s release %6.1f ms (n=%2d)  attack %6.1f ms  发声均值 %.1fdB  地板 %.1fdB"
          % (tag, np.median(rel) if rel else -1, len(rel),
             np.median(atk) if atk else -1, voiced, floor))
    return np.median(rel) if rel else -1


def main():
    src = os.environ.get("SRC", "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3")
    outdir = "/mnt/tts/diag/_tailcut"
    os.makedirs(outdir, exist_ok=True)
    print("=== 句末收尾收紧（软噪声门）===")
    for below in [12.0, 15.0, 18.0]:
        dst = os.path.join(outdir, "tc_%.0f.mp3" % below)
        f, v = process(src, dst, below)
        print(" below=%.0fdB (HI=%.1f LO=%.1f)" % (below, v - below, f + 3))
        metrics(src, "原状")
        r = metrics(dst, "处理后")
        print("     -> release %.1f ms  %s" % (r, "OK" if 15 <= r <= 45 else "偏离目标15~45ms"))
    print("\nTAILCUT_DONE")


if __name__ == "__main__":
    main()
