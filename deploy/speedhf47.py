#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""speedhf47.py —— 针对员工最新反馈的两条做客观测量：

  「滋滋滋声音存在」          → 高频能量（尤其 4~8k）与高频噪声地板
  「说话速度有点慢」          → 语速（字/秒）
  「说话一会慢一会快一点点」  → **分窗语速的稳定性**（CV / 极差）

语速不稳的量法（不依赖 ASR，纯波形）：
  按 10s 窗算「发声帧占比 × 音节核速率」太绕，这里用两个互补的：
    · voiced_rate[i] = 该窗内发声帧数 / 窗长   （停顿多少）
    · onset_rate[i]  = 该窗内能量上升沿个数 / 窗长（念了多少个字）
  两者各自算 CV（标准差/均值）与极差。CV 大 = 一会快一会慢。

用法：python3 speedhf47.py --spec spec.json
  spec: [{"name","path","chars"}, ...]   chars 用于算字/秒
"""
import argparse
import json
import subprocess

import numpy as np

SR = 16000


def load(p):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-ar", str(SR),
                        "-ac", "1", "-f", "f32le", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.float32)


def band(x, lo, hi):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    tot = X.sum()
    m = (f >= lo) & (f < hi)
    return float(X[m].sum() / tot * 100)


def hiss(x):
    """高频噪声地板：取所有 < -60dBFS 的静音帧，量它们在 4~8k 的绝对电平。
    真·滋滋声 = 静音处也有高频能量。"""
    n = int(SR * 0.02)
    m = (len(x) // n) * n
    fr = x[:m].reshape(-1, n)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    thr = np.percentile(rms, 20)          # 最安静的 20% 帧当作"静音"
    seg = fr[rms <= thr].ravel()
    if len(seg) < SR:
        return float("nan")
    return band(seg, 4000, 8000)


def onset_rate(x):
    """能量上升沿密度：近似「每秒念了几个字」。"""
    n = int(SR * 0.01)
    m = (len(x) // n) * n
    env = np.sqrt((x[:m].reshape(-1, n) ** 2).mean(axis=1))
    env = np.maximum(env, 1e-9)
    d = np.diff(20 * np.log10(env))
    # 上升沿 = 连续上升且累计 > 6dB
    up = (d > 0.6)
    cnt = 0
    run = 0.0
    for v in d:
        if v > 0.6:
            run += v
            if run >= 6.0:
                cnt += 1
                run = 0.0
        else:
            run = 0.0
    return cnt / (len(x) / SR)


def windows(x, wsec=10.0):
    n = int(SR * wsec)
    out = []
    for i in range(0, len(x) - n, n):
        out.append(x[i:i + n])
    return out


def cv(a):
    a = np.asarray(a, dtype=float)
    return float(a.std() / a.mean()) if a.mean() > 1e-9 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    a = ap.parse_args()
    items = json.load(open(a.spec, encoding="utf-8"))

    print(f"{'样本':<26}{'字/秒':>7}{'语速CV':>8}{'起音极差':>9}"
          f"{'4-8k%':>8}{'静音4-8k%':>10}{'时长':>8}")
    print("-" * 78)
    for it in items:
        x = load(it["path"])
        dur = len(x) / SR
        cps = (it.get("chars") or 0) / dur if it.get("chars") else float("nan")
        ws = windows(x)
        onsets = [onset_rate(w) for w in ws]
        hf = band(x, 4000, 8000)
        hs = hiss(x)
        print(f"{it['name']:<26}{cps:>7.2f}{cv(onsets):>8.3f}"
              f"{max(onsets)-min(onsets):>9.2f}{hf:>8.3f}"
              f"{hs:>10.3f}{dur:>8.1f}")


if __name__ == "__main__":
    main()
