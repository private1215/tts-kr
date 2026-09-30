#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""包络整形（envelope shaping）——修「咬字含糊（舌头没展开）」与「顿挫」

问题背景（2026-09-30 男声实测，1/1 对比案例原声）：
    8~16 Hz 调制能量   案例 0.1328   我们 0.0562   **-57%**
    1~4  Hz 调制能量   案例 0.3807   我们 0.5546   **+46%**

  8~16 Hz = 音节内的快速包络起伏（辅音爆破 / 送气 / 元音过渡的瞬态），
           这东西掉了就是"每个字都糊在一起、舌头没展开"。
  1~4  Hz = 0.25~1 秒尺度的响度起伏，过高就是"一顿一顿、不连贯"。
           案例的换气/断句长停顿有 29 个（≥0.45 s），我们只有 3 个。

⚠️ 为什么不能靠 `atempo` / EQ 解决：
  - atempo 只会**更**平滑瞬态（案例原声 atempo 0.873 后 8~16Hz 也掉 39%），
    而且本案走 CosyVoice3 时 speed 是模型级的，atempo 只有 0.9721（慢放 2.9%），
    实测不是主因。
  - EQ 只改频谱形状，改不了**时间维的包络**。

做法：把短时包络（10 ms 帧）按调制频率拆成 slow / mid / fast 三段，
      fast 段（≥8 Hz）做增强补咬字，slow 段只压「高于均值」的峰值去顿挫，
      再乘回波形。**不动频谱、不动时长、不引入时间拉伸伪影**。

⚠️⚠️ 两个踩过的坑（改代码前必读）：
  1. slow 段**绝不能围绕均值收窄**。第一版这么写，把停顿底噪抬起来了，
     女声 n_sent 从 22 崩到 3 —— 和 `firequalizer` 是同一个坑。
     必须 `slow - max(slow - ref, 0) * comp`，只压峰值、低谷（=停顿）不动。
  2. 静音门 `gate_db` 必须足够严（默认 -30 dB）：门外的帧 gain 恒为 1，
     保证包络操作碰不到停顿底噪。

依赖：只用到 numpy（不用 librosa），生产环境安全。
"""
import numpy as np

WIN_MS = 10.0          # 包络帧长
SLOW_HI = 4.0          # slow / mid 分界（Hz）
FAST_LO = 8.0          # mid / fast 分界（Hz）


def _frames(y, sr):
    w = max(1, int(sr * WIN_MS / 1000.0))
    n = len(y) // w
    if n < 16:
        return None, w
    e = np.sqrt((y[:n * w].reshape(n, w) ** 2).mean(axis=1) + 1e-12)
    return e, w


def _band(e, lo, hi):
    """取调制频率 [lo, hi) 的分量"""
    d = WIN_MS / 1000.0
    E = np.fft.rfft(e)
    fr = np.fft.rfftfreq(len(e), d=d)
    if lo is None:
        m = fr < hi
    elif hi is None:
        m = fr >= lo
    else:
        m = (fr >= lo) & (fr < hi)
    return np.fft.irfft(E * m, n=len(e))


def m816(y, sr, n_fft=4096):
    """8~16 Hz 调制能量占比（咬字清晰度指标）。纯 numpy，快。

    ⚠️ 必须补零到固定的 n_fft=4096：不补零的话频率分辨率随音频长度变化，
       同一个文件 51 秒版和 260 秒版会算出完全不同的数，自适应就废了。
       评估脚本（/tmp/ms.py）也是 4096，口径必须一致。
    """
    e, w = _frames(y, sr)
    if e is None:
        return 0.0
    e = e - e.mean()
    # 固定 4096：≡ 分析前 40.96 秒，音频再长也只截这么多，
    # 保证不同长度的音频频率分辨率一致（否则自适应目标没有意义）
    E = np.abs(np.fft.rfft(e * np.hanning(len(e)), n=n_fft)) ** 2
    fr = np.fft.rfftfreq(n_fft, d=WIN_MS / 1000.0)
    tot = E[(fr >= 0.5) & (fr <= 20)].sum() + 1e-12
    return float(E[(fr >= FAST_LO) & (fr < 16)].sum() / tot)


def apply(y, sr, comp=0.0, sharp=0.7, gate_db=-30.0):
    """返回 (新音频, 实际增益归一系数)。"""
    y = np.asarray(y, dtype=np.float32)
    e, w = _frames(y, sr)
    if e is None:
        return y, 1.0
    voiced = e > e.max() * (10 ** (gate_db / 20.0))
    if voiced.sum() < 16:
        return y, 1.0

    slow = _band(e, None, SLOW_HI)
    fast = _band(e, FAST_LO, None)
    mid = e - slow - fast

    # 只压 slow 的峰值，低谷（停顿）不动 —— 见文件头坑 1
    ref = slow[voiced].mean()
    slow2 = slow - np.maximum(slow - ref, 0.0) * comp
    # 放大音节内快速起伏 —— 补咬字清晰度
    fast2 = fast * (1.0 + sharp)

    e_new = np.maximum(slow2 + mid + fast2, e * 0.15)
    gain = np.where(voiced, e_new / np.maximum(e, 1e-12), 1.0)
    gain = np.clip(gain, 0.35, 3.0).astype(np.float32)

    g = np.interp(np.arange(len(y)) / w, np.arange(len(gain)), gain)
    k = np.ones(max(1, int(sr * 0.004)))
    g = np.convolve(g, k / k.sum(), mode="same").astype(np.float32)

    y2 = y * g
    r = float(np.sqrt((y ** 2).mean() + 1e-12)
              / np.sqrt((y2 ** 2).mean() + 1e-12))
    return (y2 * r).astype(np.float32), r


def fit(y, sr, target=0.12, comp=0.0, sharp_max=2.0, iters=6, gate_db=-30.0):
    """    自适应：二分求最小的 sharp，使处理后 8~16 Hz 达到 target。

    男女声差别很大（女声本来就接近案例，男声差 57%），固定强度必然有一个
    性别过度，所以必须自适应。
    返回 (新音频, 实际使用的 sharp, 处理前 8~16Hz, 处理后 8~16Hz)。
    """
    cur = m816(y, sr)
    if cur <= 0 or cur >= target:
        return y, 0.0, cur, cur
    lo, hi = 0.0, sharp_max
    best = apply(y, sr, comp, hi, gate_db)[0]
    if m816(best, sr) < target:          # 即使拉满也够不着，就用拉满
        return best, hi, cur, m816(best, sr)
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        cand, _ = apply(y, sr, comp, mid, gate_db)
        if m816(cand, sr) < target:
            lo = mid
        else:
            hi = mid
    out, _ = apply(y, sr, comp, hi, gate_db)
    return out, round(hi, 3), cur, m816(out, sr)
