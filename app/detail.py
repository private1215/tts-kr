#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补回发声区的「噪声细节」—— 治电子音 / 吃字 / 卡字（2026-10-15 第30轮）

## 为什么
员工最新反馈：**电子音、吃字、卡字**是主要问题，感情和气声已经自然。
客观指标里，频段能量分布已经和 EL 对齐（各频段 ±13% 内）、硬切 0、
真零帧 0、CER 与 EL 持平 —— **只有平坦度还有巨大偏差**：

    EL 男标杆 0.0492   裸模型 0.0345   现役成品 0.0294（融合后 0.0317）

平坦度低 = 频谱过于「尖峰 + 空谷」= 声音过于周期、缺少真实语音该有的
**送气噪声与摩擦噪声**。而送气/摩擦噪声正是韩语辅音（ㅅ/ㅊ/ㅈ/ㅋ）的载体：
    · 辅音缺噪声载体 → 被元音盖住 → **吃字**
    · 过渡处缺噪声垫底 → 衔接发硬 → **卡字**
    · 整体缺噪声细节 → 过于干净 → **电子音**
**三个症状同一个根因。**

## 做法
按**局部包络**调制地补一层粉噪（只在有声处，停顿处不受影响）：
    y' = y + noise(t) · env(t) · k
- env = 10ms RMS 平滑包络 → 噪声跟着声音走，停顿处 env≈0，**不会铺成底噪**
- k = 噪声 RMS / 局部语音 RMS。真实语音送气噪声约 −24dB → k≈0.06
- 不改时长、不改电平（后续响度闭环会把增益拉回）

## 标定（tools/detail_tune.py，融合后的 38.2s 男声）
| k | 平坦度 | b4k~8k | 停顿区电平 |
|---|---|---|---|
| 源 | 0.03166 | 1.052 | −64.7 dB |
| **0.06** | **0.04650** | **1.398** | −64.7（无泄漏）|
| 0.10 | 0.05602 | 2.004（超 EL 50%）| −64.7 |
| EL | 0.04920 | 1.336 | −62.2 |
→ k=0.06 两项都最贴合 EL，且**停顿区电平一动不动**（不会变成沙沙声）。

## 风控
- 🔴 默认 **关闭**（TTS_DETAIL_K 未设或为 0）。上线前必须过员工 A/B。
- 🔴 k 上限 0.20，超过就是明显嘶声。
"""
import os

import numpy as np

ENABLED = False
DEFAULT_K = 0.04          # 第31轮重标定值（融合后新基准）
MAX_K = 0.20              # 硬上限，防误配成嘶声
_k = os.environ.get("TTS_DETAIL_K", "").strip()
try:
    K = float(_k) if _k else 0.0
except ValueError:
    K = 0.0
if K > 0:
    K = min(K, MAX_K)
    ENABLED = True

# 🔴 带限：现役 4k~8k 已经 +3.13σ（偏多），再补宽带粉噪会把它推更高。
#    只补我们**偏低**的 2k~4k（−0.62σ）附近，抬平坦度但不恶化 4k~8k。
#    实测（tools/detail_tune2.py，494s 全篇口径）：
#      宽带 k=0.06 → flat 0.0492 但 b4k_8k 1.838（EL 1.312，超 40%）
#      带限 k=0.04 → flat 0.0408       b4k_8k 1.594（超 21%）
def _envf(name, dflt):
    v = os.environ.get(name, "").strip()
    try:
        return float(v) if v else dflt
    except ValueError:
        return dflt

LO_HZ = _envf("TTS_DETAIL_LO", 1500.0)
HI_HZ = _envf("TTS_DETAIL_HI", 4000.0)

QUIET_DB = -45.0          # 低于此电平视为停顿，不补噪声（保底，env 已天然为 0）
SEED = 20261015


def pink(n, rng):
    """粉噪，谱形接近语音长时谱。"""
    m = 1 << int(np.ceil(np.log2(max(n, 2))))
    w = np.fft.rfft(rng.standard_normal(m))
    f = np.fft.rfftfreq(m, 1.0)
    amp = np.ones_like(f)
    lo = f < 100.0
    amp[lo] = 0.5 * (1.0 - np.cos(np.pi * f[lo] / 100.0))
    hi = f > 3000.0
    amp[hi] = 0.35 * (1.0 - 0.5 * (1.0 + np.cos(
        np.pi * np.clip((f[hi] - 3000.0) / 3000.0, 0, 1))))
    return np.fft.irfft(w * amp, n=m)[:n]


def env_rms(x, sr, win_ms=10.0):
    n = max(1, int(sr * win_ms / 1000.0))
    nfr = len(x) // n
    if nfr < 1:
        return np.full(len(x), float(np.sqrt((x ** 2).mean())))
    e = np.sqrt((x[:nfr * n].reshape(nfr, n) ** 2).mean(axis=1))
    xs = (np.arange(nfr) + 0.5) * n
    return np.interp(np.arange(len(x)), xs, e, left=e[0], right=e[-1])


def filt(x, sr, lo=0.0, hi=1e9, order=4):
    """FFT 带限（线性相位，无相位失真）。

    🔴 第30轮踩过：我第一版用零极点写高通，实测是个 no-op（pink 与 hf 两列
       数值完全相同），白白浪费一轮标定。这里改成 FFT 相乘，并在
       detail_tune2.py 里打印「噪声自身频段%」自检（带限后 4k_8k 应仍占大头、
       150_300 应≈0），确认真的生效。
    """
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    g = np.ones_like(f)
    if lo > 0:
        g *= (1.0 / (1.0 + (lo / np.maximum(f, 1e-6)) ** (2 * order))) ** 0.5
    if hi < 1e8:
        g *= (1.0 / (1.0 + (np.maximum(f, 1e-6) / hi) ** (2 * order))) ** 0.5
    g[0] = 0.0
    return np.fft.irfft(X * g, n=len(x))


def apply(x, sr=24000, k=None, enabled=None, lo=None, hi=None):
    """补噪声细节。返回新数组（不改时长）。"""
    if enabled is None:
        enabled = ENABLED
    if not enabled:
        return x
    k = K if k is None else min(float(k), MAX_K)
    if k <= 0:
        return x
    lo = LO_HZ if lo is None else lo
    hi = HI_HZ if hi is None else hi
    x = np.asarray(x, dtype=np.float32)
    e = env_rms(x, sr)
    rng = np.random.default_rng(SEED)
    nz = pink(len(x), rng).astype(np.float32)
    if lo > 0 or hi < 1e8:
        nz = filt(nz, sr, lo, hi).astype(np.float32)
    nz /= (np.sqrt((nz ** 2).mean()) + 1e-20)
    # 停顿处再压一道：env 已接近 0，这里按绝对电平兜底，避免数值尾巴
    lvl = 20 * np.log10(e + 1e-12)
    gate = np.clip((lvl - QUIET_DB) / 12.0, 0.0, 1.0).astype(np.float32)
    return (x + nz * e * k * gate).astype(np.float32)
