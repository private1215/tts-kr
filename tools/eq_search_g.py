# -*- coding: utf-8 -*-
"""男女通用的 EQ 自动搜索器（基线从 JSON 读，不再把数字抄进代码）。

和 eq_search6.py 的区别：
1. 基线读 el_base_clean_{gender}.json（el_baseline_clean.py 产出），男女都能跑。
2. flat 改成**双向**目标。男声那套「只做单向护栏」是因为 EQ 只会压低 flat，
   而男声的 flat 本来就偏低，追它等于逼算法讨好一个改不了的量。
   女声相反：flat **偏高 +5.26σ**，而 EQ 正好会压低它 —— 目标与手段一致，
   所以这里 flat 就是正常的一项，权重同频段。
3. 共振峰默认关闭（F1 已被证明测量误差 > 被测量，见 el_formant_base 备注）。

用法：
    python eq_search_g.py female prod.wav --out fem_v1.wav --rounds 3
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import librosa
import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from probe_g import measure, load_base, NAME  # noqa: E402

BANDS = ["b60_150", "b150_300", "b300_500", "b500_1000",
         "b1000_2000", "b2000_4000", "b4000_8000"]
USE = BANDS + ["flat"]          # voiced 不进目标：它由停顿/语速决定，不该靠 EQ 调
FREQS = [150, 250, 350, 450, 600, 900, 1400, 2000, 3000, 4500, 6500]
GRID = [-8, -6, -4, -2, 0, 2, 4, 6, 8]
LOWPASS = 14000


def eq_str(gains):
    parts = ["equalizer=f=%d:t=q:w=1.1:g=%g" % (f, g)
             for f, g in zip(FREQS, gains) if abs(g) > 1e-9]
    parts.append("lowpass=f=%d" % LOWPASS)
    return ",".join(parts)


def apply(src, gains, dst):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-af", eq_str(gains), "-ar", "24000", "-ac", "1", dst],
                   check=True)


def cost(m, base):
    return sum(((m[k] - base[k]["mean"]) / base[k]["sd"]) ** 2 for k in USE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gender", choices=["male", "female"])
    ap.add_argument("src", nargs="+",
                    help="可传多篇，代价取各篇之和（防止单篇过拟合）")
    ap.add_argument("--out", default="")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--init", default="")
    a = ap.parse_args()

    srcs = [s if os.path.isabs(s) else os.path.join(HERE, s) for s in a.src]
    for s in srcs:
        if not os.path.exists(s):
            sys.exit("缺文件 %s" % s)
    base = load_base(a.gender)
    for k in USE:
        if k not in base:
            sys.exit("基线缺 %s" % k)

    # 🔴 多文案联合：代价 = 各篇 Σz² 之**和**。
    #    单篇搜索会过拟合——男声 v10 在 8197 上搜出 J=5.9，换到 cf9 就变成 19.2，
    #    而且弱项从 flat 变成 b4000_8000（差 5.4σ）。联合搜索逼参数去讨好两篇。
    raws = {s: measure(s) for s in srcs}
    if len(srcs) > 1:
        print("联合 %d 篇：%s" % (len(srcs), ", ".join(
            os.path.basename(s) for s in srcs)), flush=True)

    def j_of(gains):
        tot = 0.0
        for s in srcs:
            apply(s, gains, tmp)
            tot += cost(measure(tmp), base)
        return tot

    gains = [float(x) for x in a.init.split(",")] if a.init \
        else [0.0] * len(FREQS)
    assert len(gains) == len(FREQS), "init 长度应为 %d" % len(FREQS)

    tmp = os.path.join(HERE, "_eqg_tmp.wav")
    if a.init:
        best = j_of(gains)
        print("起点 J=%.1f  (init)" % best, flush=True)
    else:
        best = sum(cost(raws[s], base) for s in srcs)
        print("起点 J=%.1f  (不做 EQ)" % best, flush=True)
    raw = sum(cost(raws[s], base) for s in srcs)
    if raw < best:
        gains, best = [0.0] * len(FREQS), raw
        print("  裸片更优 J=%.1f，改用全零起点" % best, flush=True)

    for rnd in range(a.rounds):
        improved = False
        for i in range(len(FREQS)):
            cur = gains[i]
            bi, bj = cur, best
            for g in GRID:
                if abs(g - cur) < 1e-9:
                    continue
                gains[i] = g
                try:
                    j = j_of(gains)
                except Exception as e:
                    print("   FAIL", repr(e)[:70], flush=True)
                    continue
                if j < bj - 1e-9:
                    bj, bi = j, g
            gains[i] = bi
            if abs(bi - cur) > 1e-9:
                best, improved = bj, True
                print("  R%d  f=%-5d -> %+5.1f dB   J=%.1f"
                      % (rnd, FREQS[i], bi, bj), flush=True)
        print("第 %d 轮结束 J=%.1f" % (rnd, best), flush=True)
        if not improved:
            break

    for s in srcs:
        out = a.out or (os.path.splitext(s)[0] + "_eqg.wav")
        if not os.path.isabs(out):
            out = os.path.join(HERE, out)
        if len(srcs) > 1:
            out = os.path.splitext(s)[0] + "_eqg.wav"
        apply(s, gains, out)
        print("写入 %s" % out)

    print("\n最终 EQ: %s" % eq_str(gains))
    print("J = %.1f  (不做 EQ %.1f)" % (best, raw))
    for s in srcs:
        m = measure(s if not os.path.exists(
            os.path.splitext(s)[0] + "_eqg.wav") else
            os.path.splitext(s)[0] + "_eqg.wav")
        print("-- %s  Σz²=%.1f" % (os.path.basename(s), cost(m, base)))
        for k in USE:
            b = base[k]
            z = (m[k] - b["mean"]) / b["sd"]
            flag = "[X]" if abs(z) > 2 else ("[!]" if abs(z) > 1.5 else "[ok]")
            print("  %-4s %-9s 我们 %9.4f  EL %9.4f  z=%+6.2f"
                  % (flag, NAME.get(k, k), m[k], b["mean"], z))
    json.dump({"gender": a.gender, "freqs": FREQS, "gains": gains,
               "eq": eq_str(gains), "J": cost(m, base)},
              open(os.path.join(HERE, "eq_%s_result.json" % a.gender), "w"),
              ensure_ascii=False, indent=1)
    if os.path.exists(tmp):
        os.unlink(tmp)


if __name__ == "__main__":
    main()
