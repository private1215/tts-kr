#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第31轮：抬停顿底噪能不能把「发声占比」拉回 EL（离线验证）

## 假设
`probe_g` 的 `voiced` = 「10ms 帧 RMS > p30(env)×2.2 的帧占比」，是**相对阈值**。
我们的停顿填得比 EL 更安静（实测 停顿区 我们 −64.7 dB vs EL −62.2 dB）
→ p30 更低 → 阈值更低 → 更多帧被算成「发声」→ voiced 偏高。

r32 上 voiced = 0.5784 vs EL 0.5559（+3.41σ），
单这一项就贡献了 Σz² 23.9 里的 **11.63（49%）**。

## 做法
把停顿段的电平抬到 EL 实测的 −62.2 dB（只抬不够响的地方，不压响的地方），
再跑 probe_g 看 voiced / 其他项的变化。抬多少由 `--delta` 控制。

## 风控
- 只抬**低于阈值的段**，不动发声段 → 不影响语音内容
- 历史校准：−70 太轻（无贡献）、**−57 太重**（压掉原有起伏，σ 掉到 5~6）、现 −66
  → 这里只抬到 −63.5~−62.2，离 −57 很远
- 只在**离线**验证，不上线；要上线必须过员工 A/B

## 用法
    ./venv/bin/python tools/pausefloor_test.py --delta 2.5
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

SR = 24000                     # probe_g 口径
OUT_D = "/mnt/tts/diag/muffle"
SRC = "/mnt/tts/output/longprobe-r32-6cdd/longprobe-r32-6cdd-1.mp3"

try:
    import librosa
except Exception:
    librosa = None


def load(p, sr=SR):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-ac", "1",
                          "-ar", str(sr), "-f", "f32le", "-"],
                         stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def env_db(y, sr, win_ms=10.0):
    n = max(1, int(sr * win_ms / 1000.0))
    nfr = len(y) // n
    e = np.sqrt((y[:nfr * n].reshape(nfr, n) ** 2).mean(axis=1))
    return 20 * np.log10(e + 1e-12)


def raise_floor(y, sr, quiet_db=-50.0, delta_db=2.5, fade_ms=3.0):
    """只把低于 quiet_db 的帧整体抬 delta_db（带淡入淡出，不产生硬切）。"""
    n = max(1, int(sr * 0.01))
    nfr = len(y) // n
    e = np.sqrt((y[:nfr * n].reshape(nfr, n) ** 2).mean(axis=1))
    db = 20 * np.log10(e + 1e-12)
    ref = np.percentile(db[db > -200], 95) if np.any(db > -200) else 0.0
    quiet = db < (ref + quiet_db)
    # 帧级增益 → 样本级（线性插值）
    g = np.where(quiet, 10 ** (delta_db / 20.0), 1.0).astype(np.float32)
    xs = (np.arange(nfr) + 0.5) * n
    gi = np.interp(np.arange(len(y)), xs, g, left=g[0], right=g[-1])
    # 轻微平滑，避免台阶
    w = max(1, int(sr * fade_ms / 1000.0))
    k = np.hanning(w * 2 + 1)
    k /= k.sum()
    gi = np.convolve(gi, k, mode="same").astype(np.float32)
    return (y * gi).astype(np.float32)


def probe(p):
    r = subprocess.run([sys.executable, "/mnt/tts/diag/probe_g.py", "male", p],
                       capture_output=True, text=True,
                       cwd="/mnt/tts/projects/qwen3-tts")
    return (r.stdout or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=2.5)
    ap.add_argument("--src", default=SRC)
    a = ap.parse_args()

    y = load(a.src)
    print("源 %.1fs" % (len(y) / SR))
    print("\n=== 源 ===")
    print(probe(a.src))

    for d in ([a.delta] if a.delta else [1.5, 2.5, 4.0]):
        o = os.path.join(OUT_D, "pf_%.1f.wav" % d)
        sf.write(o, raise_floor(y, SR, delta_db=d), SR)
        print("\n=== 停顿底噪 +%.1f dB ===" % d)
        print(probe(o))


if __name__ == "__main__":
    main()
