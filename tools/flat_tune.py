#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""「电子音」候选修法：补回发声段的**噪声细节**（2026-10-15 第30轮）

## 为什么做这个
员工最新反馈：**「电子音、吃字、卡字」是主要问题，感情和气声已经自然**。

三大症状里，只有**平坦度**还有巨大偏差（其余全部追平甚至优于 EL）：

| 指标 | 我们（修复后男声） | EL 男标杆 | 判定 |
|---|---|---|---|
| 平坦度（probe_g 口径） | **0.0242** | **0.0461** | **−7.22σ，最大短板** |
| 硬切 / 真零帧 | 0 / 0 | 1 / 1 | ✅ 已优于 EL |
| 发声区内电平突降 >12dB | 0.63% | 0.70% | ✅ 已优于 EL |
| CER（Whisper） | 0.0994 | 0.0901 | ≈ 持平（Whisper 抓不到韩语吃字） |
| 段长中位 / 超短段占比 | 0.15s / 26.1% | 0.15s / 28.3% | ≈ 持平 |

→ **时域和内容的指标都追平了，员工还是说「吃字/卡字/电子音」。
   唯一剩下的是「声音过于周期性/过于干净」（平坦度只有 EL 的一半）。**
   发声段缺少真实语音该有的细噪声 → 辅音听着被吞（吃字）、
   过渡听着发硬（卡字）、整体发假（电子音）。
   **三个症状很可能是同一个根因。**

## 做法
按**局部包络调制**地补一层极轻的噪声（只在有声处，静音处不受影响）：
    y' = y + noise * env(t) * k
- 用**粉噪**（谱形接近语音的长时谱），不是白噪
- env 用 10ms RMS 平滑，保证噪声跟着声音走、不被听成底噪嘶声
- 目标：平坦度从 0.0242 升到 EL 的 0.0461 附近，同时 Σz² 不退化

## 用法
    ./venv/bin/python tools/flat_tune.py
"""
import json
import os
import subprocess
import sys

import numpy as np

SR16 = 16000
DIAG = "/mnt/tts/diag/muffle"
OUT = os.path.join(DIAG, "flat_tune.json")

CUR = "/mnt/tts/output/longprobe-r30fix-6cdd-2/longprobe-r30fix-6cdd-1.mp3"
EL = "/mnt/tts/data/cases/6cdd25bc75f2/merged.wav"

try:
    import librosa
except Exception:
    librosa = None


def load16(p):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-ac", "1",
                          "-ar", str(SR16), "-f", "f32le", "-"],
                         stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def flat_pg(y16):
    """完全复刻 probe_g：16k / 前 60 秒 / hop 256 / n_fft 2048。"""
    if librosa is None:
        return None
    return float(librosa.feature.spectral_flatness(
        y=y16[:60 * SR16], hop_length=256)[0].mean())


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


def env_rms(x, win_ms=10.0):
    n = max(1, int(SR16 * win_ms / 1000.0))
    nfr = len(x) // n
    e = np.sqrt((x[:nfr * n].reshape(nfr, n) ** 2).mean(axis=1))
    # 平滑到样本级
    xs = (np.arange(nfr) + 0.5) * n
    xq = np.arange(len(x))
    return np.interp(xq, xs, e, left=e[0], right=e[-1])


def add_detail(x, k, rng, win_ms=10.0, noise="pink"):
    e = env_rms(x, win_ms)
    nz = pink(len(x), rng) if noise == "pink" else rng.standard_normal(len(x))
    nz = nz / (np.sqrt((nz ** 2).mean()) + 1e-20)
    return (x + nz * e * k).astype(np.float32)


def main():
    cur = load16(CUR)
    el = load16(EL)
    f_cur, f_el = flat_pg(cur), flat_pg(el)
    print("现役平坦度 = %.5f   EL = %.5f   (差 %.1f 倍)"
          % (f_cur, f_el, f_el / f_cur))
    rows = [{"k": 0.0, "flat": f_cur, "note": "现役"}]
    print("\n%-8s %-10s" % ("k", "flat"))
    for k in (0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60):
        rng = np.random.default_rng(7)
        y = add_detail(cur, k, rng)
        f = flat_pg(y)
        rows.append({"k": k, "flat": f})
        print("%-8.2f %-10.5f %s" % (k, f, "← 接近 EL" if abs(f - f_el) < 0.004 else ""))
    json.dump({"el": f_el, "rows": rows}, open(OUT, "w"), indent=1)
    print("\n→ " + OUT)


if __name__ == "__main__":
    main()
