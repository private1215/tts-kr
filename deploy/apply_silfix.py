#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第九个方案：把句间停顿的**深度**抬到 EL 水平（−73dB → −52dB）

## 员工点名的三个位置，这次全部命中同一个特征
| 员工点名 | 实测深静音段 | 深度 | EL 同位置 |
|---|---|---|---|
| 95.5s（题1 5s）| 95.77~95.89s 125ms | **−73dB** | 最安静 −52dB |
| 131.9s（题2 1s）| 131.28~131.37s 90ms | **−240dB**（数字零）| — |
| 170.5s（题3 10s）| 169.93~170.12s 190ms | **−72dB** | 最安静 −59dB |
🔴 三个点全中，而且**都是同一个毛病：停顿太深**（比 EL 深 15~20dB，131 那处是数字零）。

旁证（silcmp.py）：静音段的数量和占比我们与 EL 相当（12 段/1.15% vs 17 段/1.03%），
**只有深度不同**（我们最低 −240dB，EL 最低 −98dB）。
zchk.py：EL 的 <−100dB 帧数 = **0**，我们 32 帧。

## 为什么"停顿太深"会被听成"突然蹦出的额/啊"
人耳对**绝对静默**极敏感。−73dB 的断口在句间出现时，
听觉系统会把它当作"中断"，并把断口**紧邻的那一两个音节凸显出来**
—— 这就是"突然冒出来的一声"，也正是"音量大小有变化"。

## 做法
把 <THR(−60dB) 且 ≥40ms 的段，抬到 TARGET(−52dB，= EL 停顿处的实测电平）。
填充用**自身房间底噪**（从本音频里挑一段真实停顿），频谱自洽；
淡入淡出 25ms。时长严格不变。
⚠️ 前八次失败都因为改的是"可听内容"（拖尾/起音/顶起幅度）；
这次改的是**本该有底噪的静音区**，方向与 EL 的实测差距一致。

用法：/mnt/tts/projects/qwen3-tts/venv/bin/python apply_silfix.py
"""
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

SR = 24000
HOP = 0.005
TMP = "/mnt/tts/diag/_silfix"
os.makedirs(TMP, exist_ok=True)
SRC = "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"
MARKS = [95.5, 99.0, 112.0, 131.9, 170.5]


def load(src, tag="x", sr=SR):
    o = os.path.join(TMP, "%s.wav" % tag)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ac", "1",
                    "-ar", str(sr), o], check=True)
    return sf.read(o, dtype="float32")[0], sr


def env_db(y, sr):
    hop = int(HOP * sr)
    n = len(y) // hop
    e = np.sqrt(np.array([(y[i * hop:(i + 1) * hop] ** 2).mean() for i in range(n)]))
    return 20 * np.log10(e + 1e-12)


def runs(m):
    d = np.diff(m.astype(np.int8))
    st = list(np.where(d == 1)[0] + 1)
    en = list(np.where(d == -1)[0] + 1)
    if m[0]:
        st.insert(0, 0)
    if m[-1]:
        en.append(len(m))
    k = min(len(st), len(en))
    return [(st[i], en[i]) for i in range(k)]


def pick_roomtone(y, sr, db, target_db):
    """从自身挑一段真实停顿当底噪源（电平接近 target）"""
    band = (db > target_db - 4) & (db < target_db + 6)
    c = [(s, e) for s, e in runs(band) if (e - s) * HOP >= 0.25]
    if not c:
        return None, None
    s, e = max(c, key=lambda t: t[1] - t[0])
    seg = y[int(s * HOP * sr):int(e * HOP * sr)]
    return seg, float(np.sqrt((seg ** 2).mean()) + 1e-12)


def pink(n, rng):
    w = rng.standard_normal(n)
    b = np.convolve(w, np.ones(64) / 64.0, mode="same")
    return b / (np.abs(b).max() + 1e-12) * 0.6


def proc(src, dst, thr_db=-60.0, target_db=-52.0, minlen_ms=40.0, mode="rtone"):
    y, sr = load(src, "src")
    db = env_db(y, sr)
    rng = np.random.default_rng(20261016)
    amp = 10.0 ** (target_db / 20.0)
    rt, rt_rms = pick_roomtone(y, sr, db, target_db) if mode == "rtone" else (None, None)
    segs = [(s, e) for s, e in runs(db < thr_db) if (e - s) * HOP * 1000 >= minlen_ms]
    n_ok = 0
    for s, e in segs:
        a, b = int(s * HOP * sr), int(e * HOP * sr)
        n = b - a
        sy = y[a:b].copy()
        if rt is not None:
            off = int(rng.integers(0, max(1, len(rt) - n)))
            nz = np.tile(rt, n // max(1, len(rt)) + 2)[off:off + n].astype(np.float32)
            nz = nz * (amp / rt_rms)
        else:
            nz = pink(n, rng) * amp
        fw = min(int(0.025 * sr), n // 2)
        if fw > 1:
            w = np.ones(n, dtype=np.float32)
            w[:fw] = np.linspace(0, 1, fw)
            w[-fw:] = np.linspace(1, 0, fw)
        else:
            w = np.ones(n, dtype=np.float32)
        # 🔴 必须是**交叉淡化**：nz*w + sy*(1-w)。
        #    只写 nz*w 是错的 —— 淡出到 0 而没把原信号混回来，
        #    等于在填补段的两端亲手挖了两个更深的坑（实测 95.5s 从 -73 掉到 -90）。
        # 🔴 保留规则也必须写死口径：只有真的还有内容（>目标+3.5dB）才留原样。
        #    绝不能写成 where(|nz| > |sy|, nz, sy) —— 那是"逐样本取较大"，
        #    粉噪过零时 |nz| 掉到 |sy| 以下 → 数字零原样漏过去，
        #    和 app/declick.py 里那个漏洞一模一样（已经踩过一次）。
        keep = np.abs(sy) > amp * 1.5
        y[a:b] = np.where(keep, sy, nz * w + sy * (1.0 - w))
        n_ok += 1
    pk = float(np.abs(y).max())
    if pk > 0.99:
        y = y * (0.99 / pk)
    sf.write(os.path.join(TMP, "o.wav"), y, sr)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", os.path.join(TMP, "o.wav"),
                    "-b:a", "192k", dst], check=True)
    return n_ok


def valleys(db):
    m = db < -60
    d = np.diff(m.astype(np.int8))
    st = list(np.where(d == 1)[0] + 1)
    en = list(np.where(d == -1)[0] + 1)
    if m[0]:
        st.insert(0, 0)
    if m[-1]:
        en.append(len(m))
    return [(s, e) for s, e in zip(st, en) if (e - s) * HOP >= 0.04]


def report(path, tag):
    y, sr = load(path, tag[:6])
    db = env_db(y, sr)
    lo = db[db < -45]
    p = np.percentile(lo, [1, 5, 10, 25, 50]) if len(lo) else [0] * 5
    vs = valleys(db)
    dep = [db[s:e].min() for s, e in vs]
    print("  %-16s 停顿 p1 %6.1f p5 %6.1f p10 %6.1f 中位 %6.1f | min %7.1f "
          "| <−100dB %3d | 深谷 %2d 段 最深 %6.1f 中位 %6.1f | RMS %6.1fdB"
          % (tag, p[0], p[1], p[2], p[4], db.min(), int((db < -100).sum()),
             len(vs), min(dep) if dep else 0,
             float(np.median(dep)) if dep else 0,
             20 * np.log10(np.sqrt((y ** 2).mean()) + 1e-12)))
    out = []
    for m in MARKS:
        a, b = int((m - 0.7) / HOP), int((m + 0.7) / HOP)
        out.append("%.1fs=%.0f" % (m, db[a:b].min()))
    print("      点名处最低：%s" % "  ".join(out))


def main():
    print("=== 修前 ===")
    report(SRC, "现役成品")
    print("参考基线：EL女声原声 停顿 p1 -90.4 p5 -60.5 p10 -58.7 中位 -51.8 | min -98.5 "
          "| <−100dB 0 | 深谷 8 段 最深 -95.8 中位 -94.5")
    for mode, thr, tgt in [("rtone", -60.0, -52.0), ("rtone", -60.0, -56.0),
                           ("rtone", -55.0, -52.0), ("pink", -60.0, -52.0)]:
        dst = os.path.join(TMP, "s_%s_%g_%g.mp3" % (mode, abs(thr), abs(tgt)))
        proc(SRC, dst, thr, tgt, 40.0, mode)
        print("\n=== 修后（%s 门%.0fdB→目标%.0fdB）===" % (mode, thr, tgt))
        report(dst, "s_%g_%g" % (abs(thr), abs(tgt)))
    print("\nSILFIX_DONE")


if __name__ == "__main__":
    main()
