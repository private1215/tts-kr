#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分频段公平对比：找「吃字」到底缺哪一段频谱（2026-10-15 第30轮）

## 假设
韩语辅音（ㅅ/ㅊ/ㅈ/ㅋ/ㅌ 等擦音、塞擦音）的能量集中在 **2k~8k**。
如果这段能量太薄 → 辅音被元音盖住 → 听感「吃字」；
过渡处能量断崖 → 「卡字」；整体缺真实噪声细节 → 「电子音」。

上一轮已证明：**EL 过 mp3 平坦度不掉，我们掉 36%** —— 我们的高频是
「一压就没」的薄层，EL 的是实打实的。这本身就是辅音层太弱的证据。

## 做法
四条信号，同口径量各频段占总能量的比例：
  我们 wav / 我们 mp3 / EL wav / EL mp3
重点看 mp3 之后每个频段掉了多少（掉得多 = 该层太薄，编码一压就没）。

## 用法
    ./venv/bin/python tools/band_fair.py
"""
import json
import os
import subprocess

import numpy as np

SR = 16000
DIAG = "/mnt/tts/diag/muffle"
OUT = os.path.join(DIAG, "band_fair.json")

SRCS = [
    ("我们 wav", os.path.join(DIAG, "t_cur.wav")),
    ("我们 mp3", "/mnt/tts/output/longprobe-r30fix-6cdd-2/"
                 "longprobe-r30fix-6cdd-1.mp3"),
    ("EL wav", os.path.join(DIAG, "el24.wav")),
    ("EL mp3", os.path.join(DIAG, "el_mp3.mp3")),
]

BANDS = [(60, 150), (150, 300), (300, 500), (500, 1000),
         (1000, 2000), (2000, 4000), (4000, 8000)]


def load16(p):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-ac", "1",
                          "-ar", str(SR), "-f", "f32le", "-"],
                         stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def bands(x):
    """各频段能量占总能量的比例（%），只用有声帧（去掉 -60dB 以下的静音）。"""
    n = 2048
    hop = 512
    nfr = (len(x) - n) // hop
    if nfr <= 0:
        return {}
    fr = x[:nfr * hop + n - hop].reshape(-1)  # placeholder
    # 手动分帧
    idx = np.arange(nfr)[:, None] * hop + np.arange(n)[None, :]
    frames = x[idx] * np.hanning(n)[None, :]
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    tot = spec.sum(axis=1)
    keep = tot > (tot.max() * 1e-6)          # 去掉数字零帧
    tot_k = tot[keep]
    freq = np.fft.rfftfreq(n, 1.0 / SR)
    out = {}
    for lo, hi in BANDS:
        m = (freq >= lo) & (freq < hi)
        out["b%d_%d" % (lo, hi)] = float(spec[keep][:, m].sum() / tot_k.sum() * 100.0)
    return out


def main():
    rows = []
    print("%-10s %s" % ("信号", "  ".join("b%d_%d" % b for b in BANDS)))
    for tag, path in SRCS:
        b = bands(load16(path))
        rows.append({"tag": tag, "path": path, "bands": b})
        print("%-10s %s" % (tag, "  ".join("%6.3f" % b["b%d_%d" % x] for x in BANDS)))

    print("\n=== mp3 造成的频段损失（相对 wav，负=被砍） ===")
    d = {r["tag"]: r["bands"] for r in rows}
    for who in ("我们", "EL"):
        print("%s:" % who)
        for lo, hi in BANDS:
            k = "b%d_%d" % (lo, hi)
            w, m = d["%s wav" % who][k], d["%s mp3" % who][k]
            print("   b%-10s wav %7.4f → mp3 %7.4f   %+6.1f%%"
                  % (k, w, m, (m / w - 1) * 100 if w else 0))

    print("\n=== 我们 mp3 vs EL mp3（员工实际听到的差距） ===")
    for lo, hi in BANDS:
        k = "b%d_%d" % (lo, hi)
        a, b_ = d["我们 mp3"][k], d["EL mp3"][k]
        print("   b%-10s 我们 %7.4f vs EL %7.4f   %+.1f%%"
              % (k, a, b_, (a / b_ - 1) * 100 if b_ else 0))

    json.dump(rows, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("\n→ " + OUT)


if __name__ == "__main__":
    main()
