# -*- coding: utf-8 -*-
"""瞬态整形（transient shaper）—— 攻「峰值因数 CF 比案例低」这个最大单项。

为什么需要它
------------
r20 五案例线上对账，唯一稳定亏损的大项是**峰值 dBFS**：

    标杆女  案例 −2.40 / 成品 −4.20   → CF 20.50 vs **18.70**（差 1.80 dB）
    标杆男  案例 −1.70 / 成品 −2.90   → CF 22.60 vs 22.00（差 0.60 dB）

CF 小 = 波形「平」= 听感上的「糊」。而它**不能用抬增益修**：响度归一会把
LUFS 拉回目标值，抬多少增益就降多少回去，1:1 抵消（r16 实测）。
也**不能用 compand 向上扩展修**：r20 试过，抬中电平会抬 RMS → 响度归一
再缩回去，峰值反而从 −2.30 掉到 −2.60/−4.30（**已证伪**）。

唯一的路子是**只抬瞬态、不抬 RMS**：让辅音爆破那几毫秒变尖，慢包络不动。

根因：神经 codec + PSOLA（PROSO）+ 包络整形都会抹瞬态。女声 F0 高、PSOLA
的基频周期短 → 重叠相加平均得更狠 → 波形更平，**所以女声 CF 亏得比男声多
3 倍**（1.80 vs 0.60 dB），这与实测完全吻合。

怎么做
------
经典 transient shaper：

    drive = clamp(fast_env − slow_env, 0, ∞) / slow_env      # 只取上升沿
    y = x · (1 + amount · drive)

fast_env 用 ~1ms 攻击（跟得上瞬态），slow_env 用 ~120ms（代表「这段有多响」）。
两者之差只在能量**上升**时为正 —— 也就是辅音起音处。增益本身再过一道 2ms
一阶平滑，避免把增益的台阶听成咔哒声。

不动频谱（纯时变增益）、不动时长、不过声码器，所以**没有相位声码器的
金属感**，也不会动 F0（A/E 不受影响）。

标定（/tmp/tshaper.py，标杆女，同一把尺子下扫 amount）
----------------------------------------------------
    amount  峰值dBFS   CF       C偏差   总分
    0.0     −4.20     18.70    6.4%    94.19
    0.1     −3.40     19.50    7.9%    95.27
    0.2     −2.40     20.50   10.0%    **95.80**   ← CF 与案例 20.50 精确相等
    0.3     −2.00     21.00   12.1%    94.55
    0.5     −1.90     21.00   15.1%    95.36
男声 CF 只差 0.60 dB，按线性外推取 amount ≈ 0.06（女声 0.20 换 1.80 dB）。

⚠️ 副作用：C 能量动态会跟着涨（瞬态也是能量）。0.20 档实测 C +0.6 dB，
   所以同一轮里必须把 dyn 的目标值往下压（见 server.py DYN_C_* 注释），
   两个改动要一起上，单独上会让 C 更偏。
"""
import numpy as np


def _follow(x, sr, atk_ms, rel_ms):
    """包络跟随器：上升快、下降慢。

    ⚠️ lfilter 做不了「系数随比较结果变」，所以两条都算出来再选：
       信号在慢包络之上时取快跟随（冲上去），否则取慢跟随（慢慢落）。
    """
    from scipy.signal import lfilter
    a_a = float(np.exp(-1.0 / (sr * atk_ms / 1000.0)))
    a_r = float(np.exp(-1.0 / (sr * rel_ms / 1000.0)))
    ax = np.abs(x)
    up = lfilter([1.0 - a_a], [1.0, -a_a], ax)
    dn = lfilter([1.0 - a_r], [1.0, -a_r], ax)
    return np.where(ax > np.concatenate(([0.0], dn[:-1])), up, dn)


def shape(x, sr, amount, atk_ms=1.0, atk_slow_ms=30.0, rel_ms=120.0,
          rel_fast_ms=15.0, max_drive=3.0, smooth_a=0.98):
    """把瞬态按 amount 抬高。amount=0 原样返回。

    ⚠️ 慢包络的**攻击**必须比快包络慢得多（30ms vs 1ms）：只有这样才能在
       能量上升沿拉开差值。标定档位（男 0.06 / 女 0.20）是按这组参数扫出来的，
       改任一参数都要重新扫描。
    """
    x = np.asarray(x, dtype=np.float32)
    if amount <= 0 or len(x) < int(0.05 * sr):
        return x
    fast = _follow(x, sr, atk_ms, rel_fast_ms)
    slow = _follow(x, sr, atk_slow_ms, rel_ms)
    drive = np.clip(fast - slow, 0.0, None) / (slow + 1e-6)
    drive = np.clip(drive, 0.0, max_drive)
    g = amount * drive
    # 增益自身平滑（τ ≈ 2ms @24k），避免增益台阶被听成咔哒声
    from scipy.signal import lfilter
    g = lfilter([1.0 - smooth_a], [1.0, -smooth_a], g)
    y = x.astype(np.float64) * (1.0 + g)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def peak_dbfs(y):
    m = float(np.max(np.abs(y))) if len(y) else 0.0
    return 20.0 * np.log10(m + 1e-12)
