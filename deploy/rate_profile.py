#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""细窗速率剖面：把「逐 60s 窗音节核速率」细化到 5s，按时间比例对齐 EL，
找出所有局部节奏异常（某段太快/太慢），用于替代「靠员工反馈逐处发现」。

用法: rate_profile.py <ours.wav> <el.wav> [win_seconds] [top_n]
输出: 每窗 ours/EL 速率与比值，按 |比值-1| 降序列出 top_n 个异常窗。
"""
import sys, subprocess, tempfile, os
import numpy as np
import soundfile as sf


def load16(path):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "16000", "-ac", "1", t], check=True)
    y, sr = sf.read(t, dtype="float32")
    os.unlink(t)
    return y, sr


def env_peaks(y, sr):
    """10ms 能量包络 + 峰检测，返回 (env, thr, peak_mask)"""
    hop = int(0.01 * sr)
    n = len(y) // hop
    e = np.sqrt(np.array([(y[i * hop:(i + 1) * hop] ** 2).mean() for i in range(n)]))
    e = np.convolve(e, np.ones(5) / 5, mode="same")
    thr = np.percentile(e, 30) * 2.2
    pk = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        if e[i] > thr and e[i] >= e[i - 1] and e[i] > e[i + 1]:
            lo = min(e[max(0, i - 8):i].min(),
                     e[i + 1:i + 9].min() if i + 9 < n else 0.0)
            if e[i] - lo > thr * 0.35:
                pk[i] = True
    return e, thr, pk


def win_rates(y, sr, win):
    e, thr, pk = env_peaks(y, sr)
    w = int(win * 100)          # 10ms 帧 -> 每窗帧数
    nw = len(e) // w
    out = []
    for k in range(nw):
        a, b = k * w, (k + 1) * w
        voiced = int((e[a:b] > thr).sum())
        peaks = int(pk[a:b].sum())
        out.append(peaks / (voiced * 0.01) if voiced > 0 else 0.0)
    return np.array(out), nw


def main():
    ours, el = sys.argv[1], sys.argv[2]
    win = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    topn = int(sys.argv[4]) if len(sys.argv) > 4 else 15

    yo, sr = load16(ours)
    ye, _ = load16(el)
    ro, nwo = win_rates(yo, sr, win)
    re_, nwe = win_rates(ye, sr, win)
    do, de = len(yo) / sr, len(ye) / sr
    print(f"ours {ours.split('/')[-1]} {do:.1f}s / {nwo} 窗 @ {win}s")
    print(f"EL   {el.split('/')[-1]} {de:.1f}s / {nwe} 窗 @ {win}s")
    print(f"时长比 EL/ours = {de/do:.4f}（用于时间比例对齐）")

    rows = []
    for k in range(nwo):
        t_o = k * win
        # 时间比例映射到 EL
        idx = int(round((t_o + win / 2) / do * nwe))
        idx = min(max(idx, 0), nwe - 1)
        if ro[k] <= 0 or re_[idx] <= 0:
            continue
        rows.append((t_o, ro[k], re_[idx], re_[idx] / ro[k]))

    print(f"\n{'窗起点':>8} {'ours':>7} {'EL':>7} {'EL/ours':>8}  判定")
    for t, a, b, r in rows:
        flag = ""
        if r > 1.15:
            flag = "← 我们偏慢"
        elif r < 0.87:
            flag = "← 我们偏快"
        print(f"{t:8.1f} {a:7.2f} {b:7.2f} {r:8.3f}  {flag}")

    bad = sorted(rows, key=lambda x: -abs(np.log(x[3])))[:topn]
    print(f"\n=== 偏离最大 {topn} 窗（|log(EL/ours)| 降序）===")
    for t, a, b, r in bad:
        print(f"  {t:7.1f}s  ours={a:6.2f}  EL={b:6.2f}  EL/ours={r:.3f} "
              f"({'我们偏快' if r < 1 else '我们偏慢'} {abs(1-r)*100:.0f}%)")

    lr = np.array([abs(np.log(x[3])) for x in rows])
    within = (np.abs(np.log(np.array([x[3] for x in rows]))) < 0.15).mean()
    print(f"\n全篇 |log比| 中位 {np.median(lr):.3f}，"
          f"落在 ±15% 内的窗占比 {within*100:.0f}%")


if __name__ == "__main__":
    main()
