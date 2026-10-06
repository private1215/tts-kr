# -*- coding: utf-8 -*-
"""能量动态扩展 —— 补「轻重音」（韵律六维里的 C 能量动态）。

为什么需要它
------------
2026-10-05 案例拆解（案例库两个 ElevenLabs 样本 vs 我们同文案输出）：

    案例-男声   seg0 C=14.09 dB   seg1 C=14.77 dB
    我们-男声   chunk1 C=12.11    chunk2 C=12.77      → 低 14~15%
    案例-女声   seg0 C=14.85      seg1 C=15.37
    我们-女声   chunk1 C=14.46    chunk2 C=13.95      → 低 3~9%

C = RMS 包络 p10~p90 的宽度，听感上就是「有没有轻重音」。我们偏低意味着
整段差不多响 —— 这正是员工说的「像读课本 / 没感情」。男声尤其明显，
而且它同时还是 A（句内音高动态）偏高的那一档：音高夸张但力度平，
合起来就是「跟吵架似的」。

怎么做
------
只改**慢包络**（50ms 窗，与 prosody_report 同口径），不动频谱、不动时长：

    db' = center + (db - center) * k

再把这个 dB 差换回增益乘到样本上。等价于一台中置的向上/向下扩展器，
慢包络相乘不会产生任何谐波或相位失真，也没有声码器重合成的音质风险。

因为映射是单调的，新的 p10/p90 正好就是旧值被拉开 k 倍，
所以 k = 目标C / 实测C 就能一步到位（增益平滑会带来一点点折损，实测为准）。

两个保护
--------
1. 静音统一处理：低于 floor 的帧先夹到 floor 再映射，
   于是所有静音帧拿到**同一个**增益 —— 停顿结构被整体搬移而不是被拉伸，
   句数 / 停顿熵（F）不会被误伤（这两维是 PBA 的活，不归它管）。
2. 增益限幅 ±9 dB 且事后做 RMS 归一 + 峰值保护，不会削波。

不做什么
--------
不改 F0（那是 A / E 的维度，需要 WORLD 重合成，风险另论）；
不改静音长度（PBA 的活）。
"""
import numpy as np


def _frames(x, sr, win=0.05, hop=0.025):
    """返回 (db, 帧中心样本位置, win, hop)。用 cumsum 算 RMS，避免 Python 循环。"""
    W = max(int(round(win * sr)), 1)
    H = max(int(round(hop * sr)), 1)
    n = (len(x) - W) // H
    if n < 8:
        return None, None, W, H
    c = np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=np.float64) ** 2)])
    s = np.arange(n) * H
    e2 = (c[s + W] - c[s]) / float(W)
    db = 20.0 * np.log10(np.sqrt(np.maximum(e2, 0.0)) + 1e-12)
    return db, s + W / 2.0, W, H


def c_of(x, sr):
    """与 prosody_report.measure 同口径的 C（只算有声帧的 p10~p90 宽度）。"""
    db, _, _, _ = _frames(x, sr)
    if db is None or len(db) < 20:
        return None
    v = db[db > np.percentile(db, 25)]
    if len(v) < 20:
        return None
    return float(np.percentile(v, 90) - np.percentile(v, 10))


def expand(x, sr, target=None, k=None, max_gain_db=9.0, smooth=3):
    """把能量动态拉宽到 target（或按给定 k）。

    返回 (y, k_used, c_before, c_after)。失败/无需处理时原样返回。
    """
    x = np.asarray(x, dtype=np.float32)
    c0 = c_of(x, sr)
    if c0 is None or c0 <= 0.5 or len(x) < sr:
        return x, 1.0, c0, c0
    if k is None:
        if not target:
            return x, 1.0, c0, c0
        k = float(target) / c0
    # ⚠️ r21 起允许 k<1（**压缩**）。旧版 clip(k,1.0,2.0) 只能扩不能压，
    #    而 r20 五案例实测成品 C 系统性**偏高** 8.2%（慢包络被后级越抬越宽），
    #    不打开这一档就没法把 C 压回案例值。压缩同样是绕 center 的单调映射，
    #    与扩展同一条公式，只是 (k-1) 变负 —— 没有新的音质风险。
    k = float(np.clip(k, 0.70, 2.0))
    if abs(k - 1.0) <= 0.002:
        return x, 1.0, c0, c0

    db, pos, W, H = _frames(x, sr)
    if db is None:
        return x, 1.0, c0, c0
    voiced = db > np.percentile(db, 25)
    center = float(np.median(db[voiced])) if voiced.any() else float(np.median(db))
    floor = float(np.percentile(db, 25))
    # 静音帧统一夹到 floor：它们拿到同一个增益，停顿结构只是整体搬移
    dbc = np.maximum(db, floor)
    g_db = np.clip((dbc - center) * (k - 1.0), -max_gain_db, max_gain_db)
    if smooth and smooth > 1:
        g_db = np.convolve(g_db, np.ones(int(smooth)) / float(smooth),
                           mode="same")
    g = (10.0 ** (g_db / 20.0)).astype(np.float64)
    gs = np.interp(np.arange(len(x), dtype=np.float64), pos.astype(np.float64), g)

    y = x.astype(np.float64) * gs
    # 整体响度回到原位（C 是宽度指标，整体平移不影响它）
    r0 = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)
    r1 = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
    y *= r0 / r1
    # 峰值保护：扩展后响的部分更响，别削波
    pk = float(np.max(np.abs(y)))
    if pk > 0.999:
        y *= 0.999 / pk
    return y.astype(np.float32), k, c0, c_of(y, sr)


if __name__ == "__main__":
    import sys
    import soundfile as sf
    for p in sys.argv[1:]:
        x, sr = sf.read(p, dtype="float32")
        if getattr(x, "ndim", 1) > 1:
            x = x.mean(axis=1)
        for tgt in (None, 14.1, 15.0):
            y, k, a, b = expand(x, sr, target=tgt, k=(1.3 if tgt is None else None))
            print(f"{p.split('/')[-1][:24]:24s} target={str(tgt):>5s} "
                  f"k={k:.3f}  C {a:.2f} -> {b if b is None else round(b, 2)}")
