# -*- coding: utf-8 -*-
"""以「EL 清洗后基线的 z 距离」为目标函数，自动搜索男声 EQ。

这一步是把优化闭环从「员工主观投票」换成「客观目标函数」的关键：
    目标 = 各频段偏离 sigma 的平方和最小
    minimize  J = sum_k z_k^2        (z 用 el_baseline2 的清洗后基线)
只要目标函数可信，机器就能自己搜，不用等员工、不受员工状态影响。

方法：坐标下降（每个频点在候选增益里搜最优，多轮直到收敛）。
为什么不用全网格：7 个频点 x 9 个增益 = 4.8M 组合，跑不完；
坐标下降对 EQ 这种近似可分的响应足够好，还能看得到每个频点的收敛过程。

用法：
    python eq_search6.py --in base_pw.wav --out eq6.wav
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
BANDS = [(60, 150), (150, 300), (300, 500), (500, 1000),
         (1000, 2000), (2000, 4000), (4000, 8000)]

# 清洗后男声基线（el_baseline2.py 产出，n=13）
BASE = {
    "b60_150": (0.0186, 0.0020), "b150_300": (0.2541, 0.0051),
    "b300_500": (0.3394, 0.0122), "b500_1000": (0.1485, 0.0076),
    "b1000_2000": (0.1488, 0.0085), "b2000_4000": (0.0766, 0.0041),
    "b4000_8000": (0.0129, 0.0006), "flat": (0.0461, 0.0030),
}
# 🔴 flat 绝不能进 EQ 的目标函数（2026-10-14 实测）：
#    EQ 只会让频谱更"结构化"，必然压低 flat；而 EL 的 flat 高（0.0461）
#    来自**底噪**，不是 EQ 能造出来的。把 flat 塞进 J，它的 z²（158）会
#    主导整个搜索，逼着算法去讨好一个它根本改不了的量，反而牺牲频段。
# 🔴 flat 不能当「优化目标」，但可以当「护栏」（2026-10-14 二次修正）：
#    把它当目标 -> 算法去讨好一个 EQ 改不了的量（EL 的高 flat 来自底噪），
#                  牺牲频段，x8197 上得到 J=13.8 但 flat 崩到 -12.57。
#    完全不管   -> PSOLA 版搜出 J=33.0 却把 flat 从 0.0419 压到 0.0021(-14.66σ)，
#                  声音过净发假，比频段偏差更致命。
#    ✅ 单向护栏：只在 flat 掉破 FLAT_MIN 时加惩罚，不奖励它变高。
NOFLAT = {"flat"}
FLAT_MIN = 0.030        # 低于此值开始罚（EL 0.0461，裸片 ~0.042）
FLAT_W = 0.5            # 惩罚权重
# 频点加密到 8 个（2026-10-14）：PSOLA 版用 7 点搜完仍有 4 项超 2σ
# （b300_500/b500_1000/b4000_8000/b2000_4000），说明固定频点不足以拟合。
# 加密区间：300~1000 补 450/800，4000~8000 补 4500/6500。
FREQS = [90, 250, 450, 800, 1400, 2500, 4500, 6500]
# 上限放到 +9：上一轮 f=90 直接顶到 +6 边界，说明 60~150Hz 缺得比预估多
GRID = [-6, -4, -2, 0, 2, 4, 6, 8]
LOWPASS = 14000


def to24(path):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "24000", "-ac", "1", t], check=True)
    y, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    return y


def measure(path):
    y = to24(path)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    f = librosa.fft_frequencies(sr=24000, n_fft=2048)
    tot = S.sum() + 1e-12
    out = {}
    for lo, hi in BANDS:
        out["b%d_%d" % (lo, hi)] = float(S[(f >= lo) & (f < hi)].sum() / tot)
    n = min(len(y), 60 * 24000)
    out["flat"] = float(librosa.feature.spectral_flatness(
        y=y[:n], hop_length=256)[0].mean())
    return out


def cost(m):
    j = sum(((m[k] - BASE[k][0]) / BASE[k][1]) ** 2
            for k in BASE if k not in NOFLAT)
    fl = m.get("flat")
    if fl is not None and fl < FLAT_MIN:
        j += FLAT_W * (((FLAT_MIN - fl) / BASE["flat"][1]) ** 2)
    return j


def eq_str(gains):
    parts = []
    for fr, g in zip(FREQS, gains):
        if abs(g) < 1e-9:
            continue
        parts.append("equalizer=f=%d:t=q:w=1.1:g=%g" % (fr, g))
    parts.append("lowpass=f=%d" % LOWPASS)
    return ",".join(parts)


def apply(src, gains, dst):
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", src, "-af", eq_str(gains),
         "-ar", "24000", "-ac", "1", dst], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="out", default="eq6.wav")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--init", default="",
                    help="初始增益，逗号分隔，长度同 FREQS")
    a = ap.parse_args()

    src = os.path.join(HERE, a.src) if not os.path.isabs(a.src) else a.src
    gains = [float(x) for x in a.init.split(",")] if a.init else [0.0] * len(FREQS)
    assert len(gains) == len(FREQS), "init 长度应为 %d" % len(FREQS)

    tmp = os.path.join(HERE, "_eq6_tmp.wav")
    best = cost(measure(src))
    print("起点 J=%.1f  (不做 EQ)" % best, flush=True)

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
                    apply(src, gains, tmp)
                    j = cost(measure(tmp))
                except Exception as e:
                    print("   FAIL", repr(e)[:80], flush=True)
                    continue
                if j < bj - 1e-9:
                    bj, bi = j, g
            gains[i] = bi
            if abs(bi - cur) > 1e-9:
                best = bj
                improved = True
                print("  R%d  f=%-5d -> %+5.1f dB   J=%.1f"
                      % (rnd, FREQS[i], bi, bj), flush=True)
        print("第 %d 轮结束 J=%.1f  gains=%s"
              % (rnd, best, ",".join("%g" % g for g in gains)), flush=True)
        if not improved:
            break

    apply(src, gains, os.path.join(HERE, a.out))
    m = measure(os.path.join(HERE, a.out))
    print("\n最终 EQ: %s" % eq_str(gains))
    print("J = %.1f" % cost(m))
    for k in BASE:
        v = m[k]
        mm, ss = BASE[k]
        z = (v - mm) / ss
        flag = "[X]" if abs(z) > 2 else ("[!]" if abs(z) > 1.5 else "[ok]")
        print("  %-4s %-12s 我们 %8.4f  EL %8.4f  z=%+6.2f"
              % (flag, k, v, mm, z))
    json.dump({"gains": gains, "freqs": FREQS, "eq": eq_str(gains),
               "J": cost(m),
               "z": {k: (m[k] - BASE[k][0]) / BASE[k][1] for k in BASE}},
              open(os.path.join(HERE, "eq6_result.json"), "w"),
              ensure_ascii=False, indent=1)
    if os.path.exists(tmp):
        os.unlink(tmp)


if __name__ == "__main__":
    sys.exit(main())
