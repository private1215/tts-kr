#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""底噪斜率的**受控**对比：同一篇成品，只换底噪，排除生成波动。

为什么必须有它
--------------
端到端「重新生成一篇」来对比是**无效实验**：VoxCPM2 同一配置连跑两次
Σz² 48.7 vs 56.7，单维度能差 ±4σ（2026-10-14 记忆第六十九节）。
实测男声斜率 -3.0 与 -1.5 两篇端到端对比：flat 只动 0.18σ（离线预测 1.7σ），
4k~8k 却差 3.26σ —— 这个差里既有生成波动也有底噪效应，**分不开**。

做法
----
线上 roomtone 是模块级 `_rng = default_rng(20261007)` 的首次调用，
用独立同 seed 实例生成同长度噪声即可**精确复现**，于是：
    无底噪版 = 成品 − 复现的噪声
    候选     = 无底噪版 + 新斜率的噪声
同一份波形只改底噪 -> 差值纯粹是斜率的效应。
"""
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, "/mnt/tts/diag")
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
from probe_g import measure, load_base          # noqa: E402
from app import roomtone                        # noqa: E402

TMPD = "/mnt/tts/diag/_rtsc"
KEYS = ("b60_150", "b150_300", "b300_500", "b500_1000",
        "b1000_2000", "b2000_4000", "b4000_8000", "flat", "voiced")
NAME = {"b60_150": "60~150", "b150_300": "150~300", "b300_500": "300~500",
        "b500_1000": "500~1k", "b1000_2000": "1k~2k", "b2000_4000": "2k~4k",
        "b4000_8000": "4k~8k", "flat": "平坦度", "voiced": "发声占比"}

# (标签, 成品路径, 成品实际用的斜率, 电平, 要对比的斜率列表)
JOBS = [
    ("男声", "/mnt/tts/diag/xf5272/prod_EL_male_8197_slopeM15.wav",
     -1.5, -56.0, (-1.5, -3.0)),
    ("男声(旧篇)", "/mnt/tts/diag/xf5272/prod_EL_male_8197_slopeM3.wav",
     -3.0, -56.0, (-3.0, -1.5)),
]


def z_of(m, base):
    zs = {}
    for k in KEYS:
        b = base[k]
        zs[k] = (m[k] - b["mean"]) / b["sd"] if b["sd"] > 1e-12 else 0.0
    return zs, sum(v ** 2 for v in zs.values())


def run(tag, path, cur_slope, db, slopes):
    base = load_base("male" if tag.startswith("男") else "female")
    prod, sr = sf.read(path, dtype="float32")
    n = len(prod)
    cur_noise = roomtone.gen_noise(n, sr, db, cur_slope,
                                   rng=np.random.default_rng(20261007))
    clean = (prod - cur_noise).astype(np.float32)
    print("=" * 78)
    print("%s  %s   %.1fs" % (tag, os.path.basename(path), n / sr))
    print("%-8s %8s %8s %8s %8s %8s" %
          ("斜率", "平坦度", "4k~8k", "150~300", "发声", "Σz²"))
    for s in slopes:
        x = clean + roomtone.gen_noise(n, sr, db, s,
                                       rng=np.random.default_rng(20261007))
        mx = float(np.max(np.abs(x)))
        if mx > 0.999:
            x = x * (0.999 / mx)
        p = os.path.join(TMPD, "c%s.wav" % str(s).replace(".", "_"))
        sf.write(p, x.astype(np.float32), sr)
        m = measure(p)
        os.unlink(p)
        z, j = z_of(m, base)
        mark = " <-成品" if abs(s - cur_slope) < 1e-9 else ""
        print("%-8s %8.5f %+8.2f %+8.2f %+8.2f %8.1f%s"
              % ("%.1f" % s, m["flat"], z["b4000_8000"], z["b150_300"],
                 z["voiced"], j, mark))
    # 顺带把全维度打出来，便于看有没有别的地方被带偏
    print("  --- 成品全维度 ---")
    m = measure(path)
    z, j = z_of(m, base)
    print("  " + "  ".join("%s %+.2f" % (NAME[k], z[k]) for k in KEYS))
    print("  Σz² = %.1f" % j)


def main():
    os.makedirs(TMPD, exist_ok=True)
    for job in JOBS:
        run(*job)


if __name__ == "__main__":
    main()
