#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""男声房间底噪**频谱斜率**实验（与女声方向相反）。

女声 flat 偏高 -> 斜率更陡（-6.0），已上线。
男声 flat **偏低**（-2.61σ，比 EL 过净）-> 需要斜率更**平**（更白），把 flat 抬回来。

前置事实（tools/rt_slope.py 女声实验已证）：
    slope 只改频谱颜色，RMS 不变 -> voiced 几乎不动（-2.44~-2.50）。
    所以这是一条**不牺牲发声占比**就能调 flat 的自由轴。

怎么做「无底噪版本」：
    现役成品 = 无底噪 + 粉噪(-56dB, -3dB/oct)，且线上是模块的第一次调用，
    用**独立的 default_rng(20261007)** 就能精确复现同一段噪声，减掉即可。
"""
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, "/mnt/tts/diag")
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
from probe_g import measure, load_base          # noqa: E402
from app import roomtone                        # noqa: E402

PROD = "/mnt/tts/diag/xf5272/prod_EL_male_8197d9ac8a35.wav"
TMPD = "/mnt/tts/diag/_rtsm"
SLOPES = (-3.0, -2.0, -1.5, -1.0, 0.0)
DB = -56.0


def main():
    os.makedirs(TMPD, exist_ok=True)
    base = load_base("male")
    prod, sr = sf.read(PROD, dtype="float32")
    n = len(prod)
    print("现役男声成品: %s  %.1fs @ %dHz" % (os.path.basename(PROD), n / sr, sr))

    # 复现线上那一次底噪（模块级 _rng 的首次调用 == 独立同 seed 首次调用）
    pink = roomtone.gen_noise(n, sr, DB, -3.0,
                              rng=np.random.default_rng(20261007))
    clean = (prod - pink).astype(np.float32)

    def score(x, tag):
        p = os.path.join(TMPD, "%s.wav" % tag)
        sf.write(p, x, sr)
        m = measure(p)
        zs = {}
        for k in ("b60_150", "b150_300", "b300_500", "b500_1000",
                  "b1000_2000", "b2000_4000", "b4000_8000", "flat", "voiced"):
            b = base[k]
            zs[k] = (m[k] - b["mean"]) / b["sd"] if b["sd"] > 1e-12 else 0.0
        os.unlink(p)
        return m, zs, sum(v ** 2 for v in zs.values())

    print()
    print("%-12s %9s %8s %8s %8s %8s" %
          ("斜率", "flat", "z(flat)", "z(voiced)", "z(150-300)", "Σz²"))
    m0, z0, j0 = score(prod, "prod")
    print("%-12s %9.5f %+8.2f %+8.2f %+8.2f %8.1f  <- 现役"
          % ("-3.0(现役)", m0["flat"], z0["flat"], z0["voiced"],
             z0["b150_300"], j0))
    for s in SLOPES[1:]:
        cand = clean + roomtone.gen_noise(n, sr, DB, s,
                                          rng=np.random.default_rng(20261007))
        mx = float(np.max(np.abs(cand)))
        if mx > 0.999:
            cand = cand * (0.999 / mx)
        m, z, j = score(cand.astype(np.float32), "s%s" % str(s).replace(".", "_"))
        print("%-12s %9.5f %+8.2f %+8.2f %+8.2f %8.1f"
              % ("%.1f dB/oct" % s, m["flat"], z["flat"], z["voiced"],
                 z["b150_300"], j))
    print()
    print("目标：flat 拉到 |z|<1.5，voiced 不掉，Σz² 比现役小")


if __name__ == "__main__":
    main()
