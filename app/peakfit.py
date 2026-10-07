# -*- coding: utf-8 -*-
"""峰值定点（peak fit）—— 把成品真峰值拉到案例的水平。

## 为什么需要它

r23 上线「停顿预算」后 CF（真峰值 − LUFS）系统性涨了 1.6 ~ 3.4 dB：

    r22-ba24c0  LUFS -24.80  真峰值 -2.70  CF 22.10
    r23-ba24c0  LUFS -24.90  真峰值 -0.90  CF 24.00
    r22-00b6a4  LUFS -22.70  真峰值 -4.00  CF 18.70
    r23-00b6a4  LUFS -22.70  真峰值 -0.60  CF 22.10

峰值偏差随之暴涨（ba24c0 6.7% -> 36.7%），按权重 0.06 折算就是 −1.8 个点，
是 r23 没能冲上 94% 的最大单项原因。

## 为什么"调限幅器"没用（实测，别再试）

ffmpeg 的 `alimiter=limit=X` 在这个版本里**是纯增益**：输入 0.771 / 0.386 /
0.154，输出一律 +2.05 dB（= 1/0.79），与"限幅"无关；只有超过 0 dBFS 才硬夹。
所以改 limit 只是整体改电平，**响度闭环会 1:1 补回来，CF 一点不变**。
CF 是信号固有的，只能靠**非线性**处理去压。

同理，直接 `x *= target/peak`（硬缩放）也是纯增益，一样会被闭环抵消。

## 它做什么

阈值 `t` 以下**完全线性**（不动），以上 tanh 渐进压向 1.0：

    y = t + (1 - t) * tanh((|x| - t) / (1 - t))        （|x| > t）

连续、无硬拐点，且只碰最响的那一小撮样本 —— 与 r21 证伪的"压慢包络"不是
一回事（那个压的是词句级轻重，会把 C 和音色一起压掉）。
`t` 由二分反解：**给定输入峰值 P 和目标峰值 Q，求出让输出峰值正好落在 Q 的 t**。

## 安全阀

`t < KNEE_MIN` 时说明目标定得比"软削能做到的"还低（再往下就是整体压扁了），
此时**放弃**，保留原样。宁可峰值不达标，也不把声音压成广播体操。
"""
import os

import numpy as np

# 默认目标由案例库按性别取中位数（见 server.peak_target）；取不到时用这个
PEAK_DB_MALE = float(os.environ.get("TTS_PEAK_DB_MALE", "-2.3"))
PEAK_DB_FEMALE = float(os.environ.get("TTS_PEAK_DB_FEMALE", "-2.4"))
# ❌ r24 已证伪，默认关：软削波把能量分布的**上端**压下来，而验收的 VAD 是
#    相对阈值 `帧电平 < p25 + 6 dB`，p25 跟着动 → 判句/停顿检出整体漂移：
#    r24-1a6c20 实测 E 3.2 -> 14.97、F 2.6 -> 12.14，净值 **-1.57 个点**
#    （peak 只赚回 +0.20）。属于"改了度量基准"而不是"改好了音频"。
#    模块留着，等哪天换成**绝对阈值**的判句口径再试。
ENABLED = os.environ.get("TTS_PEAK_FIT", "0").strip().lower() not in (
    "0", "false", "no")
# 🔴 安全阀：阈值低于此值就不再削。t=0.45 意味着 −7 dBFS 以上开始被压，
#    再低就是整体压扁，听感会毁（r21 的教训：压包络 → 音色恶化）。
KNEE_MIN = float(os.environ.get("TTS_PEAK_KNEE_MIN", "0.35"))


def _out_peak(p, t):
    """输入峰值 p、阈值 t 时，软削后的输出峰值。"""
    return t + (1.0 - t) * np.tanh((p - t) / (1.0 - t))


def solve_knee(p, q, lo=None, hi=None, iters=40):
    """反解阈值 t，使 _out_peak(p, t) == q。p < q（要抬高）时返回 None。"""
    lo = KNEE_MIN if lo is None else lo
    hi = float(p) if hi is None else hi
    if q >= p or p <= 0:
        return None
    # t 越小压得越狠、输出峰值越低。lo = KNEE_MIN 是"允许的最狠档"。
    if _out_peak(p, lo) > q:      # 连最狠档都够不到目标 → 就用最狠档，别放弃
        return lo
    if _out_peak(p, hi) < q:      # 不该发生（q < p 已在上面挡掉）
        return hi
    for _ in range(iters):
        m = 0.5 * (lo + hi)
        if _out_peak(p, m) > q:
            hi = m
        else:
            lo = m
    return 0.5 * (lo + hi)


def fit(x, target_db, knee_min=None):
    """把 x 的样本峰值软削到 target_db。**只压不抬**，返回新数组。

    取不到合适的阈值（目标太高 / 需要压得比安全阀还狠）就原样返回。
    """
    if x is None or len(x) == 0 or target_db is None:
        return x
    y = np.asarray(x, dtype=np.float32)
    p = float(np.max(np.abs(y)))
    if p <= 0:
        return x
    q = 10.0 ** (float(target_db) / 20.0)
    if p <= q:                       # 已经比目标还低，不动
        return x
    kmin = KNEE_MIN if knee_min is None else knee_min
    t = solve_knee(p, q, lo=kmin)
    if t is None:
        return x
    a = np.abs(y)
    over = a > t
    if not over.any():
        return x
    z = np.where(over, t + (1.0 - t) * np.tanh((a - t) / (1.0 - t)), a)
    return (np.sign(y) * z).astype(np.float32)


def level_for(gender, male=None, female=None):
    if str(gender or "").lower().startswith("f"):
        return PEAK_DB_FEMALE if female is None else female
    return PEAK_DB_MALE if male is None else male
