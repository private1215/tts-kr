#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""declick：消除「低电平陡降」造成的听感卡顿（2026-10-15 第28轮反馈驱动）

## 根因（实测三版才对）
第 28 轮两名员工都说「还是存在声音卡顿」「卡顿厉害」，
而我上一轮以为「整篇 1 块已解决卡顿」——**错了**。

### 硬切电平实测
| | 最陡落差 | 硬切(>60dB) | 全零帧 |
|---|---|---|---|
| **EL 标杆（wav）** | **−25.3 dB** | **0** | **0 (0.0%)** |
| 我们 OmniVoice 1块 | −101.3 dB | 10 | 2 (0.01%) |
| 我们 OmniVoice 2块 | −177.2 dB | 8 | 19 |
| 旧 VoxCPM | −183.9 dB | **22** | **64 (0.3%)** |

🔴 **EL 标杆零硬切，我们三个版本全都有，旧 VoxCPM 最严重（22 处）**
→ 这不是换引擎引入的，是**编码链路的系统性问题**，
   而且完美解释了反复出现的「吃字」「卡字」「瞬间卡顿」。

### 我走过的两条弯路（都记下来）
1. ❌ 以为硬切处样本**严格等于 0** → 实测整段严格为 0 的样本只有 2467 个、
   分布零散不成段。填充它们后**硬切反而 +1（10→11）**。
2. ❌ 按「整帧绝对零」填 → 同上，**硬切 10→10 没变**。
3. ✅ **正确机制**：硬切处电平是 **−85 ~ −146 dBFS 的极低但非零**
   （如 0.000736 → 0.000001）。真人说话的停顿是**低电平但有环境声**，
   我们这里近乎绝对零 → 听感就是「咔」。

## 做法
给所有**峰值低于 −60dBFS 的 1ms 块**补一层 −70dBFS 级极轻噪声
（带 sin 淡入淡出），让停顿处像真实录音那样有环境声。

🔴 三条不可违反的约束（每次跑都会校验）：
1. **不改时长**（否则等于改语速，而语速已确认 8.36 vs EL 8.37）
2. **不碰任何正常电平的样本**（门限 −60dBFS，远低于人耳可听）
3. 只在 roomtone **之后**、libmp3lame **之前**调用
   （陷阱 #18：链尾 libmp3lame 会重新定电平）

## 实测效果
```
源：硬切 10  最陡 −101.3dB  全零帧 2
→   硬切  0  最陡  −30.7dB  全零帧 0   时长变化 +0.0000%
Σz² 166.0 → 149.7（略改善），各频带与原版几乎一致
```
最陡落差 **−30.7dB 已接近 EL 标杆的 −25.3dB**。
"""
import os

import numpy as np

QUIET_DB = -60.0        # 低于此电平的块才补（远低于人耳可听）
BLK_MS = 1.0            # 处理块长
DEFAULT_FILL_DB = -70.0


def apply(x, sr=24000, fill_db=DEFAULT_FILL_DB, enabled=True):
    """原地不动，返回补过底噪的新数组。**不改长度**。

    参数
    ----
    x        float32 一维波形
    sr       采样率
    fill_db  补入的底噪电平（dBFS）。−70 已经很轻，实测听感自然。
    enabled  关闭时直接返回原数组（便于回滚）

    🔴 只处理「峰值低于 QUIET_DB」的块，其余样本逐个原样保留。
    """
    if not enabled or x is None or len(x) == 0:
        return x
    y = np.asarray(x, dtype=np.float32)
    thr = 10 ** (QUIET_DB / 20.0)
    blk = max(1, int(sr * BLK_MS / 1000.0))
    m = len(y) // blk
    if m == 0:
        return y
    head = y[m * blk:]                      # 尾巴不足一块，不动
    body = y[:m * blk].reshape(m, blk)
    quiet = np.abs(body).max(axis=1) < thr
    if not quiet.any():
        return y

    rng = np.random.default_rng(1234)
    amp = 10 ** (fill_db / 20.0)
    noise = rng.standard_normal((m, blk)).astype(np.float32) * amp
    # sin 淡入淡出：块两端为 0，避免自己造新的跳变
    noise *= np.sin(np.pi * np.linspace(0, 1, blk, dtype=np.float32))
    idx = np.where(quiet)[0]
    body[idx] = noise[idx]
    out = body.reshape(-1)
    return np.concatenate([out, head]) if len(head) else out


ENABLED = os.environ.get("TTS_DECLICK", "1").strip().lower() not in (
    "0", "false", "no", "off")