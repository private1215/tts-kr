# -*- coding: utf-8 -*-
"""segburst52.py —— D1 断句异声诊断（**只读，不改任何东西**）。

## 背景

员工连续 5 轮（z9/z11/z18/z19/z21）提「断句处有异声」，八条修复尝试全败。
而 `tailburst48.py` 的统计说 EL 反而更多（66.4% vs 58.1%）——
**两者测的很可能不是同一个现象**。

## 本工具的思路（与以往不同）

不再用「绝对 dB 阈值 + 统计出现率」，而是**对齐平均包络**：

1. 找出所有 >= MIN_PAUSE 的停顿（相对阈值 p35，与绝对音量无关）
2. 以每个停顿为中心，取 [前 PRE_S, 后 POST_S] 的 5ms 帧 dB 包络
3. **归一化**：把该窗口内「发声帧的 p75」对齐到 0 dB —— 消除整体音量差异
4. 所有断句窗口**叠加**，画中位 + p25/p75 带

这样文案不同也能比 —— 比的是**断句处包络的形状**，不是内容。
如果「异声」真的存在，它会表现为：
- 句末**突起**（收尾爆音 / 嘎哈）
- 句首**台阶**（起音过冲）
- 停顿**底部不平**（底噪抖动 / 地板漂移）
- 句末**跌落过快**（硬截断）

## 用法

    python3 segburst52.py 我们的.wav EL的.wav [更多...] [--out 目录]
"""
import json
import os
import sys

import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SR = 16000
FRAME = 0.005          # 5ms 帧 —— 记忆：单次 min 值/单点 dB 不可作判据，5ms 帧 RMS 才有意义
MIN_PAUSE = 0.25       # 只取「真断句」，过滤句内短隙
PRE_S, POST_S = 0.8, 0.8


def load(p, sr=SR):
    y, _ = librosa.load(p, sr=sr, mono=True)
    return y


def frame_db(y, sr=SR, frame=FRAME):
    n = int(frame * sr)
    nfr = len(y) // n
    rms = np.sqrt((y[:nfr * n].reshape(nfr, n) ** 2).mean(axis=1))
    return 20 * np.log10(rms + 1e-12), n


def find_pauses(db, frame=FRAME, min_p=MIN_PAUSE):
    voiced = db > np.percentile(db, 35)      # 相对阈值
    segs, i = [], 0
    while i < len(voiced):
        if not voiced[i]:
            j = i
            while j < len(voiced) and not voiced[j]:
                j += 1
            if i != 0 and j != len(voiced) and (j - i) * frame >= min_p:
                segs.append((i, j))
            i = j
        else:
            i += 1
    return segs


def aligned_windows(y, sr=SR):
    """返回 (矩阵, 相对时间轴秒)。每行一个断句窗口，已归一化到发声 p75 = 0dB。"""
    db, n = frame_db(y, sr)
    segs = find_pauses(db)
    pre = int(PRE_S / FRAME)
    post = int(POST_S / FRAME)
    rows = []
    for a, b in segs:
        c = (a + b) // 2                      # 以停顿中心对齐
        s, e = c - pre, c + post
        if s < 0 or e > len(db):
            continue
        w = db[s:e].copy()
        # 归一化：用窗口两端「发声部分」的 p75 作参考（避开中间的静音）
        ref = np.percentile(np.concatenate([w[:pre // 2], w[pre + pre // 2:]]), 75)
        rows.append(w - ref)
    if not rows:
        return np.zeros((0, pre + post)), (np.arange(pre + post) - pre) * FRAME
    return np.array(rows), (np.arange(pre + post) - pre) * FRAME


def main():
    argv = sys.argv[1:]
    args, outdir, i = [], "/tmp/segout52", 0
    while i < len(argv):
        a = argv[i]
        if a == "--out" and i + 1 < len(argv):
            outdir = argv[i + 1]
            i += 2
            continue
        if a.startswith("--"):
            i += 1
            continue
        args.append(a)
        i += 1
    os.makedirs(outdir, exist_ok=True)

    data = {}
    for p in args:
        print("分析", p, flush=True)
        y = load(p)
        M, t = aligned_windows(y)
        name = os.path.basename(p)
        data[name] = {"n": int(len(M)), "t": t.tolist(),
                      "med": np.median(M, axis=0).tolist() if len(M) else [],
                      "p25": np.percentile(M, 25, axis=0).tolist() if len(M) else [],
                      "p75": np.percentile(M, 75, axis=0).tolist() if len(M) else [],
                      "dur_s": round(len(y) / SR, 1)}
        print("  断句窗口 %d 个（时长 %.1fs）" % (len(M), len(y) / SR), flush=True)

    json.dump(data, open(os.path.join(outdir, "segburst.json"), "w"),
              ensure_ascii=False)

    # ---- 图 1：对齐平均包络 ----
    fig, ax = plt.subplots(figsize=(11, 5))
    for name in data:
        if not data[name]["med"]:
            continue
        t = np.array(data[name]["t"])
        med = np.array(data[name]["med"])
        p25 = np.array(data[name]["p25"])
        p75 = np.array(data[name]["p75"])
        ax.plot(t, med, lw=2, label="%s (n=%d)" % (name[:24], data[name]["n"]))
        ax.fill_between(t, p25, p75, alpha=.18)
    ax.axvline(0, color="#888", ls=":", lw=1)
    ax.set_xlabel("相对断句中心的时间 (s)   —— 负数=句末，正数=句首")
    ax.set_ylabel("相对发声电平 (dB)")
    ax.set_title("D1 断句处对齐平均能量包络（中位 + p25/p75 带）")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)
    plt.tight_layout()
    p1 = os.path.join(outdir, "seg_envelope.png")
    plt.savefig(p1, dpi=120)
    print("\n图1 -> %s" % p1)

    # ---- 图 2：关键数值对比 ----
    print("\n%-26s %8s %10s %10s %10s %10s" %
          ("文件", "断句数", "句末峰dB", "跌落dB", "停顿底dB", "句首峰dB"))
    for name in data:
        d = data[name]
        if not d["med"]:
            print("%-26s %8d  无" % (name[:26], d["n"]))
            continue
        t = np.array(d["t"])
        med = np.array(d["med"])
        pre_m = t < -0.05
        in_m = (t >= -0.05) & (t <= 0.05)
        post_m = t > 0.05
        tail_peak = med[pre_m].max()
        floor = med[in_m].min()
        head_peak = med[post_m].max()
        drop = tail_peak - floor
        print("%-26s %8d %10.1f %10.1f %10.1f %10.1f" %
              (name[:26], d["n"], tail_peak, drop, floor, head_peak))

    # ---- 图 3：单个断句样本（前 6 个，目视用）----
    for p in args:
        y = load(p)
        db, n = frame_db(y)
        segs = find_pauses(db)
        if not segs:
            continue
        pre = int(PRE_S / FRAME)
        post = int(POST_S / FRAME)
        picks = segs[:6]
        fig2, axes = plt.subplots(2, 3, figsize=(13, 6), sharey=True)
        for ax, (a, b) in zip(axes.ravel(), picks):
            c = (a + b) // 2
            s, e = max(0, c - pre), min(len(db), c + post)
            w = db[s:e]
            tt = (np.arange(len(w)) + s - c) * FRAME
            ax.plot(tt, w - np.percentile(w[:pre // 2], 75), lw=1.2, color="#1f6feb")
            ax.axvline(0, color="#888", ls=":", lw=.8)
            ax.set_title("t=%.1fs 停顿%.2fs" % (c * FRAME, (b - a) * FRAME), fontsize=8)
            ax.grid(alpha=.2)
            ax.tick_params(labelsize=7)
        plt.suptitle("D1 单样本断句包络：%s" % os.path.basename(p), fontsize=10)
        plt.tight_layout()
        p3 = os.path.join(outdir, "seg_samples_%s.png" % os.path.basename(p))
        plt.savefig(p3, dpi=110)
        print("图3 -> %s" % p3)
    print("\njson -> %s/segburst.json" % outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
