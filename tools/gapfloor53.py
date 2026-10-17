# -*- coding: utf-8 -*-
"""gapfloor53.py —— 语音段之间「间隙残留电平」对比（我们 vs EL）。

## 假设（2026-10-17 z28 员工反馈整合）

两个员工的反馈：
    员工A：「50秒有一个 0.05秒的嗯」「5秒/21秒/49秒/1分03秒 各有 0.001~0.01秒的嗯」
    员工B：「断句有异声、卡顿，男声断句情绪太差，女声有回音、卡顿」

`tools/gruntprobe53.py` 在员工标注的 101s / 143s 探测到的**不是孤立「嗯」**，
而是**语音段之间密集的 5~30ms 极短残留，峰值 -36~-44dB**（主语音 -15dB），
且这两个位置我的清理**从未碰过**（最近清理点在 85s、134s）。

→ **三者可能是同一件事**：句/词之间的间隙本该是干净静音，我们却留有
  5~30ms 的 -40dB 级残片。听起来就是「嗯」「咔哒」「卡顿」「断句异声」。

## 本工具

对每个相邻语音段之间的间隙区，统计：
    · 间隙内的最大帧电平（dB）—— 残留有多响
    · 间隙内超过 (间隙两侧电平中位 + GAP_DB) 的帧数与总时长
    · 把「有残留的间隙」占比 与 EL 对比

用法：python3 gapfloor53.py <ours> <el> [更多成对文件]
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 0.005
VOICE_DB = -40.0     # 段判定门槛（用绝对门槛，便于跨文件比较）
MIN_GAP_FRAMES = 2   # 间隙至少这么长才统计


def load16(p, sr=SR):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", str(sr), "-ac", "1", t], check=True)
    x, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def frames(x):
    n = int(FRAME * SR)
    nfr = len(x) // n
    rms = np.sqrt((x[:nfr * n].reshape(nfr, n) ** 2).mean(axis=1))
    return 20 * np.log10(rms + 1e-12)


def gaps(db):
    """返回每个间隙区 (start_frame, end_frame)。"""
    v = db > VOICE_DB
    segs, i = [], 0
    while i < len(v):
        if v[i]:
            j = i
            while j < len(v) and v[j]:
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1
    out = []
    for k in range(len(segs) - 1):
        a, b = segs[k][1], segs[k + 1][0]
        if b - a >= MIN_GAP_FRAMES:
            out.append((a, b))
    return out, segs


def stat(p):
    x = load16(p)
    db = frames(x)
    G, segs = gaps(db)
    dur = len(x) / SR
    peaks = []
    for a, b in G:
        peaks.append(float(db[a:b].max()))
    peaks = np.array(peaks) if peaks else np.array([-120.0])
    # 全局静音地板（取所有间隙区的 10 分位电平）
    allgap = np.concatenate([db[a:b] for a, b in G]) if G else np.array([-120.0])
    floor = float(np.percentile(allgap, 20))
    n_hi = int((peaks > floor + 12).sum())
    return dict(name=os.path.basename(p), dur=dur, ngap=len(G),
                pk_med=float(np.median(peaks)), pk_p90=float(np.percentile(peaks, 90)),
                floor=floor, n_hi=n_hi,
                hi_rate=n_hi / max(1, len(G)) * 100,
                hi_per_min=n_hi / dur * 60)


def main():
    fs = sys.argv[1:]
    if not fs:
        print("用法：gapfloor53.py <文件1> <文件2> ...")
        return 1
    rows = [stat(p) for p in fs]
    print("语音段间隙残留电平对比（段判定 > %.0fdB，间隙 >= %d帧=%.0fms）"
          % (VOICE_DB, MIN_GAP_FRAMES, MIN_GAP_FRAMES * FRAME * 1000))
    print()
    print("%-24s %7s %7s %9s %9s %9s %9s %11s"
          % ("文件", "时长s", "间隙数", "静音地板", "间隙峰中位", "间隙峰p90",
             "超地板12dB", "个/分钟"))
    print("-" * 96)
    for r in rows:
        print("%-24s %7.1f %7d %9.1f %9.1f %9.1f %9d %11.2f"
              % (r["name"], r["dur"], r["ngap"], r["floor"],
                 r["pk_med"], r["pk_p90"], r["n_hi"], r["hi_per_min"]))
    print()
    print("判读：『超地板12dB』= 间隙里明显高出静音底噪的残片，")
    print("      个/分钟 越高 = 句间越「脏」，对应员工说的 嗯/卡顿/断句异声。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
