# -*- coding: utf-8 -*-
"""慢尺度能量轮廓重建 —— 补「段落级的轻重」，治 AI 味。

## 为什么需要它（r25 诊断，24 条案例普查坐实）

员工反馈「ai 味比较重 / 语气不太自然」。整段均值类的 11 项指标全绿，
因为**问题不在"整体不像"，而在"内部的时间结构"**：

    句级能量慢变占比 MA10   我们 0.18~0.28   案例 0.390(男) / 0.411(女)
    句级能量 lag-1 自相关   我们 −0.19~−0.32  案例 +0.097(男) / +0.062(女)
    30 s 窗 rms 的 sd      我们 0.17~0.24 dB 案例 0.53~0.87 dB

案例 24 条里 **23 条 ac1 ≥ 0**，我们 5 条**全部 ≤ −0.18**，零重叠。
我们不是"起伏不够"，是**起伏全堆在"句到句"这个错误的尺度上、还逐句反相**
——这句响、下句就轻，十句一平均就互相抵消，听感就是「没有段落、像读课本」。

## 怎么做

逐句量**有声帧**电平 d_i（dB），拆成慢变 / 快变两部分，分别缩放再合回去：

    s = MA_W(d)                 慢变（W=10 句，约 26 秒，对应"段落"尺度）
    f = d − s                   快变（逐句）
    d' = mean + k_slow·(s − mean) + k_fast·f

两个系数**自适应反解**，不写死：

- `k_slow`：让处理后的慢变占比等于案例的 `SLOW_TARGET`
  因为 MA_W(f) ≈ 0，所以 MA_W(d') 的 sd ≈ k_slow 倍 → **一步到位，不用迭代**
- `k_fast`：在 k_slow 定下来后，补齐剩下的方差，让总 sd 等于案例的 `SD_TARGET`
  `k_fast = sqrt(max(SD_TARGET² − k_slow²·var_s, 0)) / sd_f`
  ⚠️ 只补方差，不管快变的符号 —— 我们快变是**反相**的（ac1<0），
  这是生成端固有，靠缩放改不了符号，只能靠放大慢变把它压成少数派

## 三个保护

1. **增益只作用于有声帧，静音帧恒为 1.0**。这样停顿结构、静音占比、停顿熵 F
   完全不动（它们本来就是 PBA 的活，不归它管）。
2. **均值不变**：`d'` 的均值 = `d` 的均值 → 整段 LUFS 基本不动，
   只有**相对**起伏变大，响度归一不会被推着走。
3. 增益限幅 ±`MAX_GAIN_DB`，且慢变/快变系数都有上下限。

## 顺带的意义（绕开「峰值不可修」）

记忆里「峰值/CF 不可修」的三条路都试过：抬增益被响度归一 1:1 抵消、
compand 向上扩展被缩回、瞬态整形在 mp3 上不兑现。
**共同原因是它们都抬"整条"** —— LUFS 是均值、峰值是最大值，整体抬则两者同涨，
响度归一再把均值拉回来，峰值也被拉回去了。

这里只抬**一部分句子**：LUFS（均值）不动、真峰值（最大值）由最响的那几句决定
→ **CF 真的会涨**。这是唯一合理的绕法。

不做：不改 F0（那是 proso.py 的活）、不改时长、不改频谱包络。
"""
import os
import numpy as np

# ---- 目标值：24 条案例普查的中位（tools/caseslow.py，2026-10-07）----------
# 慢变占比 MA10：男 0.390 / 女 0.411（男 min 0.298、女 min 0.289）
SLOW_TARGET_MALE = float(os.environ.get("TTS_SLOWDYN_SLOW_MALE", "0.390"))
SLOW_TARGET_FEMALE = float(os.environ.get("TTS_SLOWDYN_SLOW_FEMALE", "0.411"))
SLOW_TARGET_DEFAULT = float(os.environ.get("TTS_SLOWDYN_SLOW", "0.400"))
# 句级能量 sd (dB)：男 1.420 / 女 1.280
SD_TARGET_MALE = float(os.environ.get("TTS_SLOWDYN_SD_MALE", "1.420"))
SD_TARGET_FEMALE = float(os.environ.get("TTS_SLOWDYN_SD_FEMALE", "1.280"))
SD_TARGET_DEFAULT = float(os.environ.get("TTS_SLOWDYN_SD", "1.350"))

W = int(os.environ.get("TTS_SLOWDYN_W", "10"))          # 慢变窗（句）
K_SLOW_LO = float(os.environ.get("TTS_SLOWDYN_KSLOW_LO", "1.0"))
# 🔴 上限必须 ≥3.0：我们慢变 sd 0.198、案例 0.554 → 需要 k_slow≈2.8。
#    定 2.5 会把慢变占比卡在 0.235（实测），达不到案例的 0.390。
K_SLOW_HI = float(os.environ.get("TTS_SLOWDYN_KSLOW_HI", "3.2"))
K_FAST_LO = float(os.environ.get("TTS_SLOWDYN_KFAST_LO", "0.70"))
K_FAST_HI = float(os.environ.get("TTS_SLOWDYN_KFAST_HI", "1.50"))
MAX_GAIN_DB = float(os.environ.get("TTS_SLOWDYN_MAX_GAIN", "6.0"))
MIN_CHARS = int(os.environ.get("TTS_SLOWDYN_MIN_CHARS", "120"))
# 有声帧阈值：帧电平 > p25 + 6 dB（与 prosody_report 同源，陷阱 #9）
THR_OFF = 6.0
PCT = 25
FRAME_WIN = 0.05
FRAME_HOP = 0.025
# 增益曲线的平滑帧数（3 帧 ≈ 75 ms，兼做句边界 crossfade）
SMOOTH = 3
# 迭代次数。一次做不到目标：增益只作用于有声帧 + 帧级平滑会削掉一部分锐度，
# 实测一次只到目标占比的 81%。迭代是纯乘法、无重合成，**没有音质代价**。
#   1 次：占比 0.180→0.315，ac1 −0.278→−0.153，MA10 sd 0.198→0.442
#   2 次：占比→0.351，ac1→−0.093，MA10 sd→0.515，句级 sd 1.416（目标 1.420）
#   3 次：占比→0.369，ac1→−0.072，MA10 sd→0.557，句级 sd 1.460（略超 3%）
# 取 2：句级 sd 正好命中目标，占比进入案例 IQR(0.347~0.499)，收益递减。
ITERS = int(os.environ.get("TTS_SLOWDYN_ITERS", "2"))


def _slow_target(is_male):
    if is_male is True:
        return SLOW_TARGET_MALE
    if is_male is False:
        return SLOW_TARGET_FEMALE
    return SLOW_TARGET_DEFAULT


def _sd_target(is_male):
    if is_male is True:
        return SD_TARGET_MALE
    if is_male is False:
        return SD_TARGET_FEMALE
    return SD_TARGET_DEFAULT


def _frames(x, sr, win=FRAME_WIN, hop=FRAME_HOP):
    W_ = max(int(round(win * sr)), 1)
    H = max(int(round(hop * sr)), 1)
    n = (len(x) - W_) // H
    if n < 8:
        return None, None, W_, H
    c = np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=np.float64) ** 2)])
    s = np.arange(n) * H
    e2 = (c[s + W_] - c[s]) / float(W_)
    db = 20.0 * np.log10(np.sqrt(np.maximum(e2, 0.0)) + 1e-12)
    return db, s + W_ / 2.0, W_, H


def _ma(v, k):
    """k 点移动平均（valid），长度不足返回 None。"""
    if len(v) < k + 2:
        return None
    c = np.concatenate([[0.0], np.cumsum(v)])
    return (c[k:] - c[:-k]) / float(k)


def sent_levels(x, sr, segs):
    """逐句的**有声帧**电平 dB。返回 (dB 数组, 每句的帧索引区间)。"""
    db, pos, W_, H = _frames(x, sr)
    if db is None:
        return None, None
    thr = np.percentile(db, PCT) + THR_OFF
    voiced = db > thr
    out, spans = [], []
    for a, b in segs:
        k0 = int(a / H)
        k1 = max(k0 + 1, int(b / H))
        k1 = min(k1, len(db))
        if k1 - k0 < 3:
            continue
        v = voiced[k0:k1]
        if v.sum() < 3:
            continue
        # 只算有声帧的能量（剥掉句内静音，否则"句里停顿多"会被误读成"这句轻"）
        out.append(20.0 * np.log10(np.sqrt(np.mean(
            np.power(10.0, db[k0:k1][v] / 20.0) ** 2)) + 1e-12))
        spans.append((k0, k1))
    if len(out) < 8:
        return None, None
    return np.asarray(out, dtype=np.float64), spans


def decompose(d, w=W):
    """把句级序列拆成慢变 s / 快变 f（**逐句对齐**，长度 = len(d)）。

    🔴 反解系数和施加增益必须用**同一份** s / f，否则"理论上算出来的 k"
    和"实际施加的效果"对不上（实测偏差可达 30%）。所以这里统一用
    插值回逐句的 s_full，而不是 MA 的 valid 段。
    """
    n = len(d)
    if n < w + 2:
        return None, None
    c = np.concatenate([[0.0], np.cumsum(d)])
    idx = np.arange(n - w + 1)
    s_ma = (c[idx + w] - c[idx]) / float(w)
    centers = idx + (w - 1) / 2.0
    s = np.interp(np.arange(n, dtype=np.float64), centers, s_ma)
    f = d - s
    return s, f


def stats_of(d, w=W):
    """当前块的诊断量。⚠️ 与 decompose 同源，别用 MA 的 valid 段算。"""
    sd = float(d.std())
    if sd <= 1e-9:
        return None
    s, f = decompose(d, w)
    if s is None:
        return None
    share = float(np.std(s) / sd)
    return dict(share=share, sd_s=float(np.std(s)), sd_f=float(np.std(f)),
                sd=sd)


def solve(d, is_male=None, w=W):
    """反解 (k_slow, k_fast)。目标来自案例库普查。

    两个目标按**绝对幅度**分别对齐（案例慢变 sd 0.554 / 快变 sd 1.307），
    而不是先配占比再配总 sd —— 后者在 k_fast 顶到上限时会互相挤占，
    实测只能把占比做到 0.235 而不是 0.390。
    """
    s, f = decompose(d, w)
    if s is None:
        return 1.0, 1.0, None
    st = stats_of(d, w)
    tgt_sd = _sd_target(is_male)
    tgt_share = _slow_target(is_male)
    sd_s = float(np.std(s))
    sd_f = float(np.std(f))
    if sd_s <= 1e-9 or sd_f <= 1e-9:
        return 1.0, 1.0, st
    # 慢变：按"目标总 sd × 目标占比"反推目标慢变 sd，再除当前慢变 sd
    tgt_sd_s = tgt_sd * tgt_share
    k_slow = float(np.clip(tgt_sd_s / sd_s, K_SLOW_LO, K_SLOW_HI))
    # 快变：补齐剩下的方差（勾股，两个成分不相关）
    rest = tgt_sd ** 2 - (k_slow * sd_s) ** 2
    if rest <= 0:
        k_fast = K_FAST_LO
    else:
        k_fast = float(np.clip(np.sqrt(rest) / sd_f, K_FAST_LO, K_FAST_HI))
    return k_slow, k_fast, st


def apply(x, sr, segs=None, is_male=None, w=W, iters=ITERS):
    """重建慢尺度能量轮廓。返回 (y, k_slow, k_fast, 诊断dict)。

    ⚠️ segs 必须由**验收同源**的口径给（prosody_report.sent_windows），
    否则判句和打分对不上（陷阱 #9）。

    迭代时**每次重新判句**：增益会改变能量分布上端，而 VAD 用的是
    相对阈值 p25+6（陷阱：r24 软削波就是这么把 E/F 搞崩的）。
    """
    x = np.asarray(x, dtype=np.float32)
    y, ks, kf, diag = _once(x, sr, segs, is_male, w)
    for _ in range(max(int(iters) - 1, 0)):
        try:
            from app.prosody_report import sent_windows
            segs2, _, _ = sent_windows(y, sr)
        except Exception:
            segs2 = segs
        y, ks2, kf2, d2 = _once(y, sr, segs2, is_male, w)
        if d2 is None:
            break
        diag = d2
        ks, kf = ks2, kf2
    return y, ks, kf, diag


def _once(x, sr, segs=None, is_male=None, w=W):
    """单次重建。内部用，别直接调。"""
    x = np.asarray(x, dtype=np.float32)
    if segs is None:
        segs = [(0, len(x))]
    d, spans = sent_levels(x, sr, segs)
    if d is None:
        return x, 1.0, 1.0, None
    k_slow, k_fast, st = solve(d, is_male, w)
    if st is None or (abs(k_slow - 1.0) < 1e-3 and abs(k_fast - 1.0) < 1e-3):
        return x, 1.0, 1.0, st

    # ---- 目标句级电平：慢变放大 + 快变缩放，均值保持不变 ------------------
    # 🔴 必须与 solve() 用同一份 decompose，否则"算的 k"和"施加的 k"不一致
    s_full, f_full = decompose(d, w)
    if s_full is None:
        return x, 1.0, 1.0, st
    mean = float(d.mean())
    d_new = mean + k_slow * (s_full - mean) + k_fast * f_full
    gain_db = d_new - d
    gain_db = np.clip(gain_db, -MAX_GAIN_DB, MAX_GAIN_DB)

    # ---- 逐帧增益：只作用于有声帧，静音帧恒 1.0 --------------------------
    db, pos, W_, H = _frames(x, sr)
    if db is None:
        return x, 1.0, 1.0, st
    thr = np.percentile(db, PCT) + THR_OFF
    voiced = db > thr
    g = np.ones(len(db), dtype=np.float64)
    for gi, (k0, k1) in enumerate(spans):
        seg_v = voiced[k0:k1]
        if seg_v.sum() < 1:
            continue
        base = 10.0 ** (gain_db[gi] / 20.0)
        g[k0:k1][seg_v] = base          # 只有声帧拿增益
    # 平滑：兼做句边界 crossfade，也避免有声/静音交界处的硬跳变
    if SMOOTH > 1:
        ker = np.ones(SMOOTH, dtype=np.float64) / float(SMOOTH)
        g = np.convolve(g, ker, mode="same")

    # ---- 插值到样本级并施加 ---------------------------------------------
    n = len(x)
    gpos = np.arange(len(db)) * H + W_ / 2.0
    gs = np.interp(np.arange(n, dtype=np.float64), gpos, g,
                   left=g[0], right=g[-1])
    y = x * gs.astype(np.float32)
    # 保险：别把整体电平推走（均值不变是设计目标，这里兜个底）
    r0 = float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))
    r1 = float(np.sqrt(np.mean(np.asarray(y, dtype=np.float64) ** 2)))
    if r1 > 1e-9 and r0 > 1e-9:
        y = (y * (r0 / r1)).astype(np.float32)
    diag = dict(stats=st, k_slow=k_slow, k_fast=k_fast,
                gain_min=float(gain_db.min()), gain_max=float(gain_db.max()))
    return y.astype(np.float32, copy=False), k_slow, k_fast, diag
