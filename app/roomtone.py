# -*- coding: utf-8 -*-
"""房间底噪（room tone）—— 修「停顿处是绝对静音」造成的瞬间停顿与电流声。

为什么需要它
------------
2026-10-07 用户反馈 r22-6cdd25 / r22-8197d9 **开头几秒有电流声和瞬间停顿**。
逐 100ms 电平实测（前 8 秒）：

    我们 r22-6cdd25 : -128  -77  -27 ...    句中停顿 -173 -174 -158
    我们 r22-8197d9 : -122  -74  -30 ...    句中停顿 -164 -163 -105
    案例 6cdd25     :  -36  -19  -16 ...    句中停顿  -49  -50  -55
    案例 8197d9     :  -35  -17  -13 ...    句中停顿  -51  -55  -51

帧电平分位数（0.05s 帧，dBFS）：

    |        | p5    | p10   | p15   | p25   | 语音(p75) |
    | 案例男 | -61.5 | -53.2 | -48.9 | -38.8 | -24.4    |
    | 案例女 | -52.0 | -45.3 | -41.7 | -36.6 | -21.8    |
    | 我们   |-160.7 | -89.3 | -77.2 | -44.9 | -24.2    |

🔴 我们的静音是 **−160 dB 的数字真零**（模型输出就是零，不是"安静"）。后果有两个：
1. **瞬间停顿**：语音 −24 dB ↔ 静音 −160 dB，0.1 秒内跳 130 dB。人耳在"有"和
   "完全没有"之间硬切会觉得突兀、像卡了一下；案例是 −50 dB 的房间声，
   听感是自然的呼吸间隙。
2. **电流声**：低电平段经过 `libmp3lame` 时比特分配极少，量化噪声的相对失真很大；
   再加上链路里的 aexciter / 空气感 EQ 在近零段做非线性处理，会留下杂散产物，
   解码后就是"滋滋"的电流声。

为什么铺底噪**不会**把评分搞坏
------------------------------
验收用的 VAD 是**相对阈值** `帧电平 < p25 + 6 dB`。底噪把静音帧从 −160 抬到 −56，
p25 会被抬一点点，但语音帧（−24）仍在阈值之上、静音帧仍在之下 → 分类不变。
（记忆 §5 r16 已验证：铺底噪不改静音占比/C/F。本轮上线前会再实测一次确认。）

频谱形状
--------
用**粉红噪声**（-3 dB/oct），物理上就是房间底噪的典型形状，且与案例实测的
底噪谱形（低频最强、5kHz 约 −14 dB、7kHz 约 −16 dB）吻合在几 dB 内。
6 kHz 以上额外 −6 dB/oct，避免听起来发嘶。

电平取在案例的 p5 与 p10 之间（男 −56 / 女 −50）：再低就没有"房间感"，
再高会在语音段落里被听见。
"""
import os

import numpy as np

# 电平（dBFS）。男声案例 p5=-61.5 / p10=-53.2；女声 p5=-52.0 / p10=-45.3
ROOMTONE_DB_MALE = float(os.environ.get("TTS_ROOMTONE_DB_MALE", "-56"))
ROOMTONE_DB_FEMALE = float(os.environ.get("TTS_ROOMTONE_DB_FEMALE", "-50"))
ENABLED = os.environ.get("TTS_ROOMTONE", "1").strip().lower() not in (
    "0", "false", "no")
# 6 kHz 以上额外滚降（dB/oct）
HF_EXTRA_ROLLOFF = 6.0
HF_KNEE = 6000.0
# 低频地板：再往下没有意义，且会带进隆隆声
F_FLOOR = 60.0

_rng = np.random.default_rng(20261007)


def gen_noise(n, sr, db, rng=None):
    """生成 n 采样、RMS = db dBFS 的粉红噪声（FFT 塑形，比级联滤波快且准）。

    幅度谱 = 粉红(功率 −3 dB/oct) × 6 kHz 以上额外 −6 dB/oct。
    """
    g = rng or _rng
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    # 一次生成整段的 2 的幂长度，避免分块拼接处的相关性
    m = 1 << int(np.ceil(np.log2(max(n, sr))))
    W = np.fft.rfft(g.standard_normal(m))
    f = np.fft.rfftfreq(m, 1.0 / sr)
    ff = np.maximum(f, F_FLOOR)
    amp = (F_FLOOR / ff) ** 0.5                       # 粉红：功率 -3 dB/oct
    over = np.log2(np.maximum(f, HF_KNEE) / HF_KNEE)  # 超过拐点的倍频程数
    amp = amp * 10.0 ** (-(HF_EXTRA_ROLLOFF / 20.0) * np.maximum(over, 0.0))
    y = np.fft.irfft(W * amp, n=m).astype(np.float32)[:n]
    rms = np.sqrt(np.mean(y ** 2)) + 1e-20
    return (y * (10.0 ** (db / 20.0) / rms)).astype(np.float32)


def apply(x, sr=24000, db=-56.0, rng=None):
    """把房间底噪叠到信号上。**原地不改**，返回新数组。

    ⚠️ 必须在**响度归一之后**叠：叠在归一之前会把 RMS 抬一点点，
    闭环会整体缩回，等于白铺（还会把响度搞偏）。
    """
    if not ENABLED or x is None or len(x) == 0:
        return x
    y = np.asarray(x, dtype=np.float32)
    n = y + gen_noise(len(y), sr, db, rng=rng)
    # 绝不因为这一步把峰值顶过 0 dBFS
    m = float(np.max(np.abs(n))) if len(n) else 0.0
    if m > 0.999 and m > 0:
        n = n * (0.999 / m)
    return n.astype(np.float32)


def level_for(gender):
    """按性别取电平。识别不出就按男声（更保守、更不容易听出噪声）。"""
    if str(gender or "").lower().startswith("f"):
        return ROOMTONE_DB_FEMALE
    return ROOMTONE_DB_MALE
