# -*- coding: utf-8 -*-
"""韵律 A / E 的 F0 重映射 —— 用 Praat PSOLA 把「句内动态」和「句尾多样性」
拉到案例水平。

为什么单独做这一块：
    dyn.py 只能改能量（韵律 C），改不了音高。而六维里我们最差的两项
    恰好都是 F0 维度的：
        女声 E 句尾多样性 24.1 vs 案例 34.4（-30%）：听感就是「一个调」
        男声 A 句内动态   9.29 vs 案例 8.36（+11%）：听感就是「跟吵架似的」
    两者方向相反，必须分性别、分维度独立缩放。

核心是同一个单调映射（与 dyn.py 对能量做的完全同源）：
        semi' = center + (semi - center) * k
    ⇒ p15/p85 宽度、以及 semi 对时间的斜率，都**精确乘 k**。
    所以 k 可以一步算出来：k = 目标 / 当前，不需要迭代搜索。

    A：整句绕「句内 F0 中位」缩放 k_a
    E：句尾窗口再叠加一层「绕同一中位」的缩放 k_tail，权重从 0 渐变到 1，
       这样既有斜率放大，又不会在句尾出现音高跳变。

只改 F0，不改时长、不改频谱包络 —— 所以 D（语速 CV）、F（停顿熵）、
C（能量动态）都不受影响（PSOLA 重合成保持原长度与原包络）。

⚠️ 重合成有音质代价（Praat PSOLA 的固有人造感），所以：
    - 缩放限幅 k ∈ [0.75, 1.35]，超出就只走到边界；
    - 必须先离线试算 + 耳朵验收再决定上不上线。
"""
import numpy as np

try:
    import parselmouth
    from parselmouth.praat import call as _pcall
except Exception:                                  # pragma: no cover
    parselmouth = None
    _pcall = None

K_LO, K_HI = 0.75, 1.35          # 单次缩放限幅
TAIL_SEC = 0.35                  # 与 prosody_report 的 E 窗口一致
STEP = 0.01                      # PitchTier 时间步长（秒）


def _fminmax(is_male):
    """韩语男女的基频搜索范围。判不出性别时放宽。"""
    if is_male is True:
        return 60.0, 300.0
    if is_male is False:
        return 90.0, 450.0
    return 55.0, 500.0


def tier_of(x, sr, is_male=None, step=STEP):
    """取 F0 曲线。返回 (times, f0)；失败返回 (None, None)。"""
    if parselmouth is None:
        return None, None
    fmin, fmax = _fminmax(is_male)
    snd = parselmouth.Sound(np.asarray(x, dtype=np.float64), float(sr))
    manip = _pcall(snd, "To Manipulation", step, fmin, fmax)
    pt = _pcall(manip, "Extract pitch tier")
    n = _pcall(pt, "Get number of points")
    if n < 8:
        return None, None
    t = np.array([_pcall(pt, "Get time from index", i) for i in range(1, n + 1)])
    f = np.array([_pcall(pt, "Get value at index", i) for i in range(1, n + 1)])
    ok = f > 0
    return t[ok], f[ok]


def _resynth(x, sr, t, f, is_male=None, step=STEP):
    """用新的 (t, f) 重建 PitchTier 并 PSOLA 重合成。"""
    fmin, fmax = _fminmax(is_male)
    snd = parselmouth.Sound(np.asarray(x, dtype=np.float64), float(sr))
    manip = _pcall(snd, "To Manipulation", step, fmin, fmax)
    pt = _pcall(manip, "Extract pitch tier")
    _pcall(pt, "Remove points between", 0.0, snd.get_total_duration() + 1.0)
    for i in range(len(t)):
        _pcall(pt, "Add point", float(t[i]), float(f[i]))
    _pcall([manip, pt], "Replace pitch tier")
    y = _pcall(manip, "Get resynthesis (overlap-add)")
    return np.asarray(y.values[0], dtype=x.dtype)


def shape(t, f, segs_sec, k_a=1.0, k_tail=1.0, tail_sec=TAIL_SEC):
    """按句缩放 F0。返回新的 f 数组。

    segs_sec: [(start_sec, end_sec), ...]，来自 prosody_report.sent_windows

    A：整句内绕该句 F0 中位缩放 k_a                ⇒ A 精确 × k_a
    E：句尾 tail_sec 窗口再叠一层，权重 0→1 渐变    ⇒ 句尾斜率再 × k_tail
    """
    g = f.copy()
    if abs(k_a - 1.0) < 1e-4 and abs(k_tail - 1.0) < 1e-4:
        return g
    semi = 12.0 * np.log2(np.maximum(f, 1e-6))
    out = semi.copy()
    # 每个 F0 点归属哪一句（句外的点不动）
    starts = np.array([s for s, _ in segs_sec]) if segs_sec else np.zeros(0)
    ends = np.array([e for _, e in segs_sec]) if segs_sec else np.zeros(0)
    for si, (s0, e0) in enumerate(segs_sec):
        m = (t >= s0) & (t < e0)
        if m.sum() < 4:
            continue
        c = np.median(semi[m])                       # 句中位（半音域）
        w_a = np.ones(m.sum())
        if abs(k_tail - 1.0) > 1e-4 and e0 - s0 > 0.2:
            tw = min(tail_sec, (e0 - s0) * 0.6)
            # 句尾窗口内权重 0→1 平滑上升，避免音高跳变
            w_a = np.clip((t[m] - (e0 - tw)) / tw, 0.0, 1.0)
            w_a = w_a * w_a * (3.0 - 2.0 * w_a)      # smoothstep
        kk = k_a * (k_tail ** w_a)
        out[m] = c + (semi[m] - c) * kk
    _ = starts, ends, si                              # 占位：保留结构便于扩展
    return np.power(2.0, out / 12.0)


def ae_of(x, sr, is_male=None, segs=None, tail_sec=TAIL_SEC):
    """用 Praat F0 口径量 A / E。

    ⚠️ 必须和「目标值」同一口径：目标值是用同一套公式在案例 merged.wav 上
    量出来的（男 A=9.495/E=45.493，女 A=9.498/E=33.511），与
    prosody_report 的 librosa.pyin 口径**数值不同但方向一致**。
    只要 cur 和 target 同源，k = target/cur 就是对的。

    为什么不用 prosody_report.measure：pyin 几分钟音频要 1~2 分钟，
    放进每个任务太慢；Praat 只要 1~2 秒。
    """
    t, f = tier_of(x, sr, is_male=is_male)
    if t is None:
        return None, None
    if segs is None:
        segs = [(0, len(x))]
    segs_sec = [(s / float(sr), e / float(sr)) for s, e in segs]
    per, sl = [], []
    for s0, e0 in segs_sec[:80]:
        m = (t >= s0) & (t < e0)
        if m.sum() >= 5:
            ff = f[m]
            per.append(12.0 * np.log2(np.percentile(ff, 85) /
                                      np.percentile(ff, 15) + 1e-12))
        a = max(s0, e0 - tail_sec)
        m2 = (t >= a) & (t < e0)
        if e0 - a >= 0.15 and m2.sum() >= 5:
            ff = f[m2]
            ff = ff[(ff > np.percentile(ff, 5)) & (ff < np.percentile(ff, 95))]
            if len(ff) >= 5:
                semi = 12.0 * np.log2(ff / np.median(ff) + 1e-12)
                sl.append(np.polyfit(np.arange(len(semi)), semi, 1)[0] *
                          (sr / 256.0))
    A = float(np.median(per)) if len(per) >= 2 else None
    E = float(np.std(sl)) if len(sl) >= 2 else None
    return A, E


def _clip(k):
    return float(np.clip(k, K_LO, K_HI))


def fit(x, sr, target_a=None, target_e=None, cur_a=None, cur_e=None,
        is_male=None, segs=None, tail_sec=TAIL_SEC, max_semi_shift=4.0):
    """把 A / E 拉到目标值。

    cur_a / cur_e：当前实测值（由 prosody_report 量出来，必须同口径）。
    只在给定的时候才动对应的维度。
    返回 (y, k_a_used, k_tail_used)；失败原样返回。
    """
    if parselmouth is None:
        return x, 1.0, 1.0
    if (target_a is None or cur_a is None or cur_a <= 0) and \
       (target_e is None or cur_e is None or cur_e <= 0):
        return x, 1.0, 1.0
    t, f = tier_of(x, sr, is_male=is_male)
    if t is None:
        return x, 1.0, 1.0
    if segs is None:
        segs = [(0.0, len(x) / float(sr))]
    segs_sec = [(s / float(sr), e / float(sr)) for s, e in segs]

    k_a = _clip(target_a / cur_a) if (target_a and cur_a and cur_a > 0) else 1.0
    # E 会被 k_a 顺带乘一次，所以尾部缩放要把它除掉
    if target_e and cur_e and cur_e > 0:
        k_tail = _clip(target_e / (cur_e * k_a))
    else:
        k_tail = 1.0
    g = shape(t, f, segs_sec, k_a=k_a, k_tail=k_tail, tail_sec=tail_sec)
    # 绝对音高整体位移限幅（缩放绕中位，理论上不该跑，兜个底）
    med_o, med_n = np.median(f), np.median(g)
    shift = 12.0 * np.log2(med_n / max(med_o, 1e-6))
    if abs(shift) > max_semi_shift:
        g = g * np.power(2.0, -(shift - np.sign(shift) * max_semi_shift) / 12.0)
    try:
        y = _resynth(x, sr, t, g, is_male=is_male)
    except Exception:
        return x, 1.0, 1.0
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    y = y[:len(x)]
    return y.astype(x.dtype, copy=False), k_a, k_tail
