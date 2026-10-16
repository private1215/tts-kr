#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第六个方案：**连续平滑增益**压顶起（取代分段压制，消除"音量大小有变化"）

## 为什么推翻分段压制（z43 三票全 D 的教训）
一次性压到「上下文+3dB」客观指标极好（Σz² 32.1→22.6、平坦度 +3.83σ→+0.20σ），
但员工 디아 的自由文本点破了副作用：**「断句有异声。音量大小有变化」**
—— 分段压制在段边界留下 20ms 的音量台阶，人耳听成"忽大忽小"。
而且邻位会相对冒头（95.5s 压掉后 96.6s 冒出 +15dB）。

迭代压制更糟：顶起 25.1→6.9 处/分是"达标"了，但
**Σz² 32.1→72.0、发声占比 −2.00σ→−5.79σ** —— 第二轮压的已经是正常语音。

## 本方案
不做"分段 + 硬边"，改成**处处定义的连续增益**：
    e(t)  = 5ms RMS 包络（dB）
    m(t)  = e(t) 的 1.5s 窗 75 分位（"本该有的响度水平"）
    g(t)  = min(0, m(t) − e(t))   只衰减、不提升，并**限制最大衰减**
    g     = 100ms 移动平均平滑 → 不存在台阶
    out   = y × 10^(g/20)
一个样本一个增益值，天然连续，不会产生"音量跳变"。

## 闸门
① 顶起密度应下降（EL 8.7 / 我们 25.1 处每分）
② Σz² 不许比 32.1 差，平坦度向 0 靠（最好落进 ±3σ）
③ 包络平滑度不能变差（这是 z43 翻车的直接原因）

用法：/mnt/tts/projects/qwen3-tts/venv/bin/python apply_smooth.py
      KB=1.5 MAXCUT=8 等环境变量可调
"""
import importlib.util
import os
import subprocess
import sys

import numpy as np
import soundfile as sf
from scipy.ndimage import percentile_filter

SR = 24000
HOP = 0.005
TMP = "/mnt/tts/diag/_smooth"
os.makedirs(TMP, exist_ok=True)
SRC = "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"
MARKS = [("题1 5s", 95.5), ("题2 1s", 131.9), ("题3 10s", 170.5)]


def _ab():
    spec = importlib.util.spec_from_file_location("ab", "/mnt/tts/diag/apply_bump.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["ab"] = m
    spec.loader.exec_module(m)
    return m


AB = _ab()


def load(src, tag="x"):
    out = os.path.join(TMP, "%s.wav" % tag)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ar", str(SR),
                    "-ac", "1", out], check=True)
    return sf.read(out, dtype="float32")


def env_db(y, sr):
    hop = int(HOP * sr)
    n = len(y) // hop
    e = np.sqrt(np.array([(y[i * hop:(i + 1) * hop] ** 2).mean() for i in range(n)]))
    return 20 * np.log10(e + 1e-12)


def smooth1d(x, ms):
    w = max(1, int(ms / 1000.0 / HOP))
    k = np.ones(w) / w
    pad = np.pad(x, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(pad, k, mode="valid")[:len(x)]


def roughness(db):
    """包络粗糙度：一阶差分的标准差（越大 = 音量起伏越突兀）"""
    return float(np.std(np.diff(db)))


def process(src, dst, kb=1.5, q=75.0, maxcut=8.0, smooth_ms=100.0):
    y, sr = load(src)
    db = env_db(y, sr)
    win = max(3, int(kb / HOP))
    m = percentile_filter(db, q, size=win, mode="nearest")
    g = np.minimum(0.0, m - db)                 # 只衰减
    g = np.maximum(g, -maxcut)                  # 限制最大衰减
    g = smooth1d(g, smooth_ms)
    gi = np.interp(np.arange(len(y)) / float(sr), np.arange(len(db)) * HOP, g)
    out = y * (10.0 ** (gi / 20.0)).astype(np.float32)
    pk = float(np.abs(out).max())
    if pk > 0.99:
        out = out * (0.99 / pk)
    sf.write(os.path.join(TMP, "o.wav"), out, sr)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", os.path.join(TMP, "o.wav"),
                    "-b:a", "192k", dst], check=True)
    return float(np.mean(g)), float(np.percentile(g, 1))


def report(path, tag):
    y, sr = AB.load(path, tag[:4])
    db = env_db(y, sr)
    b, fl, vo, ns = AB.find_bumps(y, sr, 8.0, maxlen=0.35, ctx_win=0.4)
    dur = len(y) / sr
    res = []
    for nm, mk in MARKS:
        near = [t for t, l, lv, c in b if abs(t - mk) < 2.0]
        res.append("%s=%s" % (nm, "%.1fs" % near[0] if near else "消除"))
    print("  %-22s 顶起%3d处=%5.1f/分  包络粗糙度 %.3f  %s"
          % (tag, len(b), len(b) / (dur / 60), roughness(db), " ".join(res)))


def main():
    print("=== 连续平滑增益 ===")
    report(SRC, "现役")
    for kb, q, mc in [(1.5, 75.0, 8.0), (1.5, 60.0, 8.0), (1.0, 70.0, 6.0), (2.0, 70.0, 10.0)]:
        dst = os.path.join(TMP, "s_%g_%g_%g.mp3" % (kb, q, mc))
        mg, p1 = process(SRC, dst, kb, q, mc)
        print("kb=%.1f q=%g maxcut=%gdB  平均衰减 %.2fdB 最深 %.1fdB"
              % (kb, q, mc, mg, p1))
        report(dst, "  -> kb%g/q%g/mc%g" % (kb, q, mc))
    print("\nSMOOTH_DONE")


if __name__ == "__main__":
    main()
