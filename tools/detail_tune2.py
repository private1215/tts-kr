#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第31轮：在**融合后的新基准**上重新标定补细节的 k（全篇口径）

## 为什么重新标定
融合补丁把 496s 成品的平坦度从 0.0242 拉到 **0.0373**（EL 0.0461），
基准变了，第30轮按旧基准定的 k=0.06 会**过头**。

## 🔴 口径铁律
- `probe_g` 只测**前 60s**，且历史教训是「短片 4k~8k 系统性低 40~60%，
  拿短片调高频会得到相反结论」→ 所以：
  · 平坦度同时报 **前60s**（与 probe_g 对齐）和 **全篇**（更可信）
  · 4k~8k 频段**只报全篇**
- 对照组 EL 用同一口径、同一长度。

## 新增：带限噪声
现役 4k~8k 已经 **+3.13σ（偏多）**，再补宽带粉噪会把它推更高。
所以加一档 **1.5k~4k 带限噪声**：只补我们**偏低**的 2k~4k（−0.62σ），
既抬平坦度又不恶化 4k~8k。

## 用法
    ./venv/bin/python tools/detail_tune2.py
"""
import json
import os
import subprocess

import numpy as np

SR = 16000
OUT = "/mnt/tts/diag/muffle/detail_tune2.json"
NEW = "/mnt/tts/output/longprobe-r31-6cdd/longprobe-r31-6cdd-1.mp3"
EL = "/mnt/tts/diag/muffle/el24.wav"

try:
    import librosa
except Exception:
    librosa = None


def load16(p, t=None):
    c = ["ffmpeg", "-v", "error", "-i", p, "-ac", "1", "-ar", str(SR)]
    if t:
        c += ["-t", str(t)]
    c += ["-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(c, stdout=subprocess.PIPE,
                                        check=True).stdout,
                         dtype=np.float32).copy()


def flat(y, sec=None):
    yy = y[:int(sec * SR)] if sec else y
    return float(librosa.feature.spectral_flatness(
        y=yy, hop_length=256)[0].mean())


def band(y):
    n, hop = 2048, 512
    nfr = (len(y) - n) // hop
    idx = np.arange(nfr)[:, None] * hop + np.arange(n)[None, :]
    fr = y[idx] * np.hanning(n)[None, :]
    sp = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1.0 / SR)
    tot = sp.sum()
    return {lo: float(sp[:, (f >= lo) & (f < hi)].sum() / tot * 100)
            for lo, hi in [(150, 300), (2000, 4000), (4000, 8000)]}


def pink(n, rng):
    m = 1 << int(np.ceil(np.log2(max(n, 2))))
    w = np.fft.rfft(rng.standard_normal(m))
    f = np.fft.rfftfreq(m, 1.0)
    amp = np.ones_like(f)
    lo = f < 100.0
    amp[lo] = 0.5 * (1.0 - np.cos(np.pi * f[lo] / 100.0))
    hi = f > 3000.0
    amp[hi] = 0.35 * (1.0 - 0.5 * (1.0 + np.cos(
        np.pi * np.clip((f[hi] - 3000.0) / 3000.0, 0, 1))))
    return np.fft.irfft(w * amp, n=m)[:n]


def filt(x, sr, lo=0.0, hi=1e9, order=4):
    """FFT 带限（线性相位）。lo/hi 单位 Hz。"""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    g = np.ones_like(f)
    if lo > 0:
        g *= (1.0 / (1.0 + (lo / np.maximum(f, 1e-6)) ** (2 * order))) ** 0.5
    if hi < 1e8:
        g *= (1.0 / (1.0 + (np.maximum(f, 1e-6) / hi) ** (2 * order))) ** 0.5
    g[0] = 0.0
    return np.fft.irfft(X * g, n=len(x))


def env_rms(x, win_ms=10.0):
    n = max(1, int(SR * win_ms / 1000.0))
    nfr = len(x) // n
    e = np.sqrt((x[:nfr * n].reshape(nfr, n) ** 2).mean(axis=1))
    xs = (np.arange(nfr) + 0.5) * n
    return np.interp(np.arange(len(x)), xs, e, left=e[0], right=e[-1])


def add(x, k, rng, lo=0.0, hi=1e9):
    e = env_rms(x)
    nz = pink(len(x), rng)
    if lo > 0 or hi < 1e8:
        nz = filt(nz, SR, lo, hi)
        # 自检：确认带限真的生效（第30轮 FFT 高通是个 no-op，白测了一轮）
        b = band(nz.astype(np.float32))
    else:
        b = None
    nz = nz / (np.sqrt((nz ** 2).mean()) + 1e-20)
    return (x + nz.astype(np.float32) * e * k).astype(np.float32), b


def main():
    src = load16(NEW)
    el = load16(EL, t=len(src) / SR)
    print("我们 %.1fs   EL(同长) %.1fs" % (len(src) / SR, len(el) / SR))

    f_el60, f_el_full = flat(el, 60), flat(el)
    b_el = band(el)
    f060, f0full = flat(src, 60), flat(src)
    b0 = band(src)
    print("\n%-14s %9s %9s | %8s %8s %8s"
          % ("", "flat前60s", "flat全篇", "b150_300", "b2k_4k", "b4k_8k"))
    print("%-14s %9.5f %9.5f | %8.3f %8.3f %8.3f"
          % ("EL", f_el60, f_el_full, b_el[150], b_el[2000], b_el[4000]))
    print("%-14s %9.5f %9.5f | %8.3f %8.3f %8.3f"
          % ("我们(融合后)", f060, f0full, b0[150], b0[2000], b0[4000]))

    rows = []
    print("\n%-10s %-6s %9s %9s | %8s %8s %8s"
          % ("噪声", "k", "flat前60s", "flat全篇", "b150_300", "b2k_4k", "b4k_8k"))
    for tag, lo, hi in (("宽带粉噪", 0.0, 1e9), ("带限1.5k-4k", 1500.0, 4000.0)):
        for k in (0.02, 0.04, 0.06, 0.09):
            rng = np.random.default_rng(7)
            y, nb = add(src, k, rng, lo, hi)
            b = band(y)
            r = {"tag": tag, "k": k, "flat60": flat(y, 60),
                 "flat_full": flat(y), "bands": b}
            rows.append(r)
            print("%-10s %-6.2f %9.5f %9.5f | %8.3f %8.3f %8.3f"
                  % (tag, k, r["flat60"], r["flat_full"],
                     b[150], b[2000], b[4000]))
            if nb and k == 0.02:
                print("             (噪声自身频段%%: 150_300=%.2f 2k_4k=%.2f "
                      "4k_8k=%.2f)" % (nb[150], nb[2000], nb[4000]))

    # 选：flat前60s 最接近 EL，且 4k_8k 不超过 EL 太多
    cand = [r for r in rows if r["bands"][4000] <= b_el[4000] * 1.25]
    best = min(cand, key=lambda r: abs(r["flat60"] - f_el60)) if cand else None
    print("\n推荐（4k~8k 不超过 EL 的 1.25 倍，且 flat 最接近）：")
    if best:
        print("   %s k=%.2f → flat60 %.5f（EL %.5f） b4k_8k %.3f（EL %.3f）"
              % (best["tag"], best["k"], best["flat60"], f_el60,
                 best["bands"][4000], b_el[4000]))
    json.dump({"el": {"flat60": f_el60, "flat_full": f_el_full, "bands": b_el},
               "src": {"flat60": f060, "flat_full": f0full, "bands": b0},
               "rows": rows}, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("→ " + OUT)


if __name__ == "__main__":
    main()
