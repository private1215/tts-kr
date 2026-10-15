#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""declick：消除「数字零」造成的听感卡顿（2026-10-15 第28轮反馈驱动，第30轮定案）

## 根因（实测三版才对）
第 28 轮两名员工都说「还是存在声音卡顿」「卡顿厉害」，
而我上一轮以为「整篇 1 块已解决卡顿」——**错了**。

### 🔴🔴 第30轮：真正的量化证据（此前一直被 σ 误导）
按 10ms 帧统计「静音帧」（<−45dB）的电平分布：

| | 静音帧占比 | <−90dB | **<−200dB（真零）** | 中位 |
|---|---|---|---|---|
| 我们（处理前） | 7639 帧 | 1.165% | **54 帧 (0.707%)** | −51.8dB |
| **EL 标杆** | 11819 帧 | 3.638% | **1 帧 (0.008%)** | −56.6dB |

→ **我们的「数字零」帧是 EL 的 88 倍**。这才是「咔」的真身。
→ 之前算出的 σ=22.29（vs EL 13.21）**完全是这 54 个 −300dB 帧撑出来的**，
   不是什么"起伏不足"。σ 是被污染的指标，别再拿它当目标。

### 我走过的弯路（都记下来，别重踩）
1. ❌ 以为硬切处样本**严格等于 0** → 整段严格为 0 的样本零散不成段，
   填充后**硬切反而 +1（10→11）**。（因为按 1ms 块切，装不下）
2. ❌ 按「整帧绝对零」填 → 同上，硬切 10→10 没变。
3. ✅ 硬切处电平是 **−85 ~ −146 dBFS 的极低但非零**，必须按**连续段**处理。
4. ❌ 第29轮用**整段替换**（`out[a:b] = tone`）→ 硬切虽清零，
   但把本来有起伏的停顿一起盖掉了（σ 被压到 6.97）。
5. ❌ 以为要把填充电平抬到 −66dB 才能"做出起伏" → 错。
   **EL 最深的静音帧也只到 −89dB**（p5=−89.04）；抬到 −66 是反向操作。
6. ❌ 起伏不能用慢调制：慢周期 0.55~2.9s，而停顿段中位只有 **0.07s**，装不下。

## 做法（第30轮定案）
**逐样本「幅度下限」** —— 只在原信号低于填充电平时才顶上来：
    out = where(|噪声| > |原信号|, 噪声, 原信号)
- 门限 `QUIET_DB=−100`：只认真正的数字零，**不动** −100dB 以上任何内容
- 填充 `FILL_DB=−89`：**对齐 EL 的 p5（−89.04dB）**，绝不过量
- 粉噪（1/f）+ 1ms sin 淡入淡出

🔴 三条不可违反的约束（每次跑都会校验）：
1. **不改时长**（否则等于改语速，而语速已确认 8.36 vs EL 8.37）
2. **不碰任何 ≥−100dBFS 的样本**（远低于人耳可听）
3. 只在 roomtone **之后**、libmp3lame **之前**调用
   （陷阱 #18：链尾 libmp3lame 会重新定电平）

## 实测效果（553.6s 女声长篇，离线）
```
             硬切   最陡5帧落差   真零帧   σ    Σz²(女声基线)
处理前         8     −218.89dB    54    22.29   70.9
现役(整段替换)  0     −23.81dB     0     6.97   73.4   ← 盖掉太多
本版(幅度下限)  0     −33.96dB     0     8.94   72.7   ← 定案
EL标杆         1     −55.86dB     1    13.21   (自评 6.8)
```
Σz² 三版差异 <4%（在噪声内），**无回退**；真零帧 54→0 是唯一确定收益。
"""
import os

import numpy as np

# 🔴 第30轮：门限提到 −100dB —— 只认真正的数字零。
#    −72/−85/−95 都会把 EL 本来就有的 −89dB 深静音一起处理掉。
#    实测：q=−72 时 σ 只有 6.55，q=−100 时 8.94，q=−110 与 −100 无差别
#    （说明 −100 以下基本没有"有效信号"了，再严也没得保）。
QUIET_DB = -100.0
BLK_MS = 1.0            # 处理块长（仅文档用，实际按连续段）

# 🔴 第30轮：填充电平 −89dB = **EL 静音帧的 p5（−89.04dB）**。
#    原则：我们永远不该比 EL 最深的静音还安静，也不该比它响。
#    第29轮的 −66dB 比 EL 深静音高 23dB，属于反向操作（把 σ 压到 6.97）。
DEFAULT_FILL_DB = -89.0

# 停顿段**内部**起伏的实测参考（仅文档，本参数现由电平/噪声类型自然决定）：
#      EL 标杆(6cdd/8197)  7.26 / 7.30 dB
#      旧 VoxCPM           7.35 dB   ← 与 EL 一致
#      OmniVoice           2.62 dB   ← 掉到 1/3
TARGET_WOBBLE_DB = 13.2

# 填充模式：floor = 逐样本幅度下限（定案）
#            replace = 第29轮的整段替换（已否决，盖掉太多）
DEFAULT_MODE = os.environ.get("TTS_DECLICK_MODE", "floor").strip().lower()

# EL 停顿的实测谱形（作为合成目标）：中高频为主、高频几乎没有
#   0-100:11%  100-300:10%  300-1k:23%  1-3k:54%  3-8k:1%
# → 典型「远处环境声」，不是白噪声。


def _pink(n, rng):
    """1/f 粉噪：真实环境噪声的频谱形状，比白噪声自然。"""
    m = 1 << int(np.ceil(np.log2(max(n, 2))))
    w = np.fft.rfft(rng.standard_normal(m))
    f = np.fft.rfftfreq(m, 1.0)
    amp = np.ones_like(f)
    lo = f < 100.0
    amp[lo] = 0.5 * (1.0 - np.cos(np.pi * f[lo] / 100.0))
    hi = f > 3000.0
    amp[hi] = 0.35 * (1.0 - 0.5 * (1.0 + np.cos(
        np.pi * np.clip((f[hi] - 3000.0) / 3000.0, 0, 1))))
    y = np.fft.irfft(w * amp, n=m)
    return y[:n]


def _room_tone(n, sr, rng, fill_db):
    """生成环境声：宽带粉噪，电平按 fill_db（RMS 口径）对齐。

    🔴 不需要"慢调制"或"段内起伏"：
       EL 的 σ 来自**帧与帧之间电平随机跳动**（11325 帧总分布宽度 43dB），
       不是段内慢漂移。宽带噪声天然就是这个统计特性。
    """
    base = _pink(n, rng)
    base /= (np.abs(base).max() + 1e-20)
    rms0 = np.sqrt((base ** 2).mean()) + 1e-20
    return (base * (10 ** (fill_db / 20.0) / rms0)).astype(np.float32)


def apply(x, sr=24000, fill_db=DEFAULT_FILL_DB, enabled=True,
          mode=DEFAULT_MODE, quiet_db=QUIET_DB):
    """返回补过环境声的新数组。**不改长度**。

    🔴🔴 第30轮的关键修正：**逐样本幅度下限，不做整段替换**。
       整段替换（`out[a:b] = tone`）会把本来 −75dB、有自然起伏的停顿
       一律盖成 −66dB 的恒定噪声 → 起伏被抹平（σ 6.97）。
       正确做法：`where(|tone| > |x|, tone, x)`
       —— 原信号比噪声响的地方**一个字节都不动**，
          只有掉到数字零的地方才被顶上来。
    """
    if not enabled or x is None or len(x) == 0:
        return x
    y = np.asarray(x, dtype=np.float32)
    thr = 10 ** (quiet_db / 20.0)
    quiet = np.abs(y) < thr
    if not quiet.any():
        return y

    rng = np.random.default_rng(1234)
    out = y.copy()
    min_len = int(sr * 0.002)          # <2ms 的不算停顿，那是信号过零
    edges = np.diff(quiet.astype(np.int8))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if quiet[0]:
        starts.insert(0, 0)
    if quiet[-1]:
        ends.append(len(y))

    ramp = max(1, int(sr * 0.001))      # 1ms 淡入淡出
    for s, e in zip(starts, ends):
        a, b = int(s), int(e)
        if b - a < min_len:
            continue
        tone = _room_tone(b - a, sr, rng, fill_db)
        rl = min(ramp, (b - a) // 2)
        env_r = np.ones(b - a, dtype=np.float32)
        if rl > 0:
            env_r[:rl] = np.sin(np.pi * np.linspace(0, 1, rl,
                                                     dtype=np.float32))
            env_r[-rl:] = np.sin(np.pi * np.linspace(1, 0, rl,
                                                      dtype=np.float32))
        tone = tone * env_r
        if mode == "replace":
            out[a:b] = tone
        else:                            # floor：逐样本幅度下限
            seg = out[a:b]
            out[a:b] = np.where(np.abs(tone) > np.abs(seg), tone, seg)
    return out


ENABLED = os.environ.get("TTS_DECLICK", "1").strip().lower() not in (
    "0", "false", "no", "off")
