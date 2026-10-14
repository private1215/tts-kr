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
# 🔴 护栏改成「掉破 EL 的 1.5σ 下界就罚」（2026-10-14 三次修正）：
#    原来是 FLAT_MIN=0.030（约 -5.4σ）太松——v10 现役 flat=0.0356(-3.51σ)
#    是当前**最差的一项**，却是"合规"的，搜索完全不管它。
#    验收口径是「所有维度落在 EL 1.5σ 内」（EL 原文 8197 自己也有 ±1.5σ 抖动），
#    所以护栏就设在同一个门槛上：EL 0.0461 - 1.5*0.0030 = 0.0416。
FLAT_MIN = 0.0416
FLAT_W = 1.0            # 与频段项同量纲（都是 sigma^2），不再打折

# 🎯 共振峰目标（2026-10-14）：**用 EL 真实分布，不再拍脑袋**。
#    el_formant_base.py 实测 EL 男声 n=13：F1 412.3±5.1 / F2 1556.2±8.6
#    （之前手拍的 SD=(25,90) 松了 5~10 倍，共振峰项在 J 里约等于 0，白加。）
#    换成真实 sigma 后，共振峰与 7 个频段项**同量纲**（都是「偏离几个 EL sigma」），
#    FORMANT_W 保持 1.0 即可，不需要人为调权重。
#    实测：v10 的 F2 偏 +3.8% = **+6.9σ**，是全部维度里最大的一项，
#    这就是员工三轮都说「像感冒」的量化证据（共振峰偏高=声道变短）。
FORMANT_TARGET = (412.3, 1556.2)
FORMANT_SD = (5.1, 8.6)        # el_formant_male.json 实测，勿手改
# 🔴🔴 F1 必须剔出目标函数（2026-10-14 晚，formant_stability.py 实测推翻）：
#    同一个文件只改 Praat 参数/换分析段，F1 抖动 **46~200Hz**，
#    段内前后半段自身就差 23Hz —— 而 EL 的自然 sigma 只有 **5.1Hz**。
#    **测量误差 >> 被测量**，F1 的 z 是噪声，放进去只会把搜索带偏。
#    F2 稍稳（固定 maxf=5000/n=5/ts=0.01 时段内差 ~16Hz，EL sigma 8.6Hz），
#    勉强可用，但**不能再当"最大偏差项"**——之前说「F2 +3σ = 感冒音」是过度解读。
FORMANT_USE = (0.0, 1.0)       # 逐维权重：F1 关闭、F2 启用
FORMANT_W = 1.0                # 总开关，0 = 完全不看共振峰
FORMANT_JSON = "el_formant_male.json"

_p = os.path.join(HERE, FORMANT_JSON)
if os.path.exists(_p):
    _j = json.load(open(_p, encoding="utf-8"))
    FORMANT_TARGET = (_j["F1"]["mean"], _j["F2"]["mean"])
    FORMANT_SD = (_j["F1"]["sd"], _j["F2"]["sd"])
    print("共振峰基线来自 %s：F1 %.1f±%.1f  F2 %.1f±%.1f (n=%d)"
          % (_p, FORMANT_TARGET[0], FORMANT_SD[0],
             FORMANT_TARGET[1], FORMANT_SD[1], _j["n"]), flush=True)
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


def load16(path):
    """降采样到 16k（flat 必须 16k 口径，与 EL 基线一致）。"""
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "16000", "-ac", "1", t], check=True)
    y, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    return y, 16000


def formants(path):
    """F1/F2 中位（Praat Burg 16k）。

    🔴 口径必须和 el_formant_base.py 完全一致：取 [10% 时长, +180s]。
       之前这里是绝对 [60,150s]，跟基线的相对窗口不是一段，
       「频段分布随文本位置变化」同样适用于共振峰，口径不一致就不可比。
    """
    import parselmouth
    pr = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True)
    dur = float(pr.stdout.strip() or 0)
    start = max(0.0, 0.10 * dur)
    length = min(180.0, max(5.0, dur - start))
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error",
                    "-ss", "%.2f" % start, "-t", "%.2f" % length,
                    "-i", path, "-ar", "16000", "-ac", "1", t], check=True)
    s = parselmouth.Sound(t)
    os.unlink(t)
    pit = s.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=400)
    fm = s.to_formant_burg(time_step=0.01, max_number_of_formants=5,
                           maximum_formant=5000)
    ts = np.arange(0.05, s.duration - 0.05, 0.01)
    F1, F2 = [], []
    for tt in ts:
        f0 = pit.get_value_at_time(tt)
        if f0 is None or np.isnan(f0) or f0 <= 0:
            continue
        a = fm.get_value_at_time(1, tt)
        b = fm.get_value_at_time(2, tt)
        if a is None or b is None or np.isnan(a) or np.isnan(b) \
                or a <= 0 or b <= 0:
            continue
        F1.append(a)
        F2.append(b)
    if len(F1) < 100:
        return None, None
    return float(np.median(F1)), float(np.median(F2))


def measure(path):
    y = to24(path)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    f = librosa.fft_frequencies(sr=24000, n_fft=2048)
    tot = S.sum() + 1e-12
    out = {}
    for lo, hi in BANDS:
        out["b%d_%d" % (lo, hi)] = float(S[(f >= lo) & (f < hi)].sum() / tot)
    y16, _ = load16(path)
    out["flat"] = float(librosa.feature.spectral_flatness(
        y=y16[:60 * 16000], hop_length=256)[0].mean())
    if FORMANT_W > 0:
        f1, f2 = formants(path)
        out["F1"], out["F2"] = f1, f2
    return out


def cost(m):
    j = sum(((m[k] - BASE[k][0]) / BASE[k][1]) ** 2
            for k in BASE if k not in NOFLAT)
    fl = m.get("flat")
    if fl is not None and fl < FLAT_MIN:
        j += FLAT_W * (((FLAT_MIN - fl) / BASE["flat"][1]) ** 2)
    if FORMANT_W > 0 and m.get("F1") and m.get("F2"):
        j += FORMANT_W * sum(
            FORMANT_USE[i] * ((m[nm] - FORMANT_TARGET[i]) / FORMANT_SD[i]) ** 2
            for i, nm in enumerate(("F1", "F2")))
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
    # 🔴 必须先在 init 上量一次（2026-10-14 修）：
    #    之前 best = cost(不加 EQ 的原始片)，--init 传进来的现役 EQ 从来没被评估过，
    #    贪心是从零点重走的。结果 v11 搜出 J=33.0，反而比起点 v10 的 24.6 更差，
    #    而脚本还"以为"自己一直在变好。现在：先量 init，再和裸片取较优者。
    if a.init:
        apply(src, gains, tmp)
        best = cost(measure(tmp))
        print("起点 J=%.1f  (init=%s)" % (best, a.init), flush=True)
    else:
        best = cost(measure(src))
        print("起点 J=%.1f  (不做 EQ)" % best, flush=True)
    _raw = cost(measure(src))
    if _raw < best:
        gains, best = [0.0] * len(FREQS), _raw
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
    if m.get("F1"):
        for i, nm in enumerate(("F1", "F2")):
            z = (m[nm] - FORMANT_TARGET[i]) / FORMANT_SD[i]
            flag = "[X]" if abs(z) > 2 else ("[!]" if abs(z) > 1.5 else "[ok]")
            print("  %-4s %-12s 我们 %8.1f  EL %8.1f  (%+5.1f%%)  z=%+6.2f%s"
                  % (flag, nm, m[nm], FORMANT_TARGET[i],
                     100 * (m[nm] / FORMANT_TARGET[i] - 1), z,
                     "  (不入J)" if FORMANT_USE[i] == 0 else ""))
    zs = {k: (m[k] - BASE[k][0]) / BASE[k][1] for k in BASE}
    if m.get("F1"):
        for i, nm in enumerate(("F1", "F2")):
            if FORMANT_USE[i] != 0:
                zs[nm] = (m[nm] - FORMANT_TARGET[i]) / FORMANT_SD[i]
    zs["_max"] = max(abs(v) for v in zs.values())
    json.dump({"gains": gains, "freqs": FREQS, "eq": eq_str(gains),
               "J": cost(m), "z": zs},
              open(os.path.join(HERE, "eq6_result.json"), "w"),
              ensure_ascii=False, indent=1)
    print("\n最差偏离 max|z| = %.2f  (%s)"
          % (zs["_max"], max((k for k in zs if k != "_max"),
                             key=lambda k: abs(zs[k]))))
    if os.path.exists(tmp):
        os.unlink(tmp)


if __name__ == "__main__":
    sys.exit(main())
