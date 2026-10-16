#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""silfloor：句间停顿「地板抬起」——让停顿像 EL 一样平躺在房间底噪上

（2026-10-16 第44轮，员工第 3 次点名「断句有异声 / 突然出来短音」后定案）

## 根因（同文案逐秒对齐后测出，不是猜的）
我们这个 198.6s 女声成品与 EL 女声原声 52727ec43154 是**同一份文案**，
且 95s / 131s / 170s / 183s 四处逐字对齐（EL 只领先 0.3~0.6s）→ 绝对秒可直接对照。

| | EL 女声原声 | 我们现役 |
|---|---|---|
| 停顿帧中位 | −51.8dB | −51.9dB ← **一样** |
| p5 | −60.5dB | **−73.5dB** ← 尾巴深 13dB |
| p10 | −58.7dB | −62.4dB |
| min | −98.5dB | **−240dB**（数字零） |
| <−100dB 帧 | **0** | 32 |
| 深谷(<−60dB,≥40ms) | 8 段 | 20 段（中位 −76.7dB） |

→ EL 的停顿「衰减到 −52dB 就**平躺**在房间底噪上」；
  我们的停顿「穿过 −52dB 继续**往下沉**」，沉到 −73dB，甚至沉成数字零。
  沉下去后语音再起来，落差从 ~30dB 变成 50~220dB —— 听感上第一个音节
  **「突然蹦出来」**，也就是员工说的「突然出来短音 / 额 / 啊」。

## 员工点名 3 处 ↔ 深静音谷，3/3 精确命中
z43 题1 切自 90s 起，员工说「5 秒里有额」→ 95.50s（−73dB、125ms）✅
z43 题2 切自 130s 起，员工说「1 秒里有啊」→ 131.28s（**数字零**、75ms）✅
z43 题3 切自 160s 起，员工说「10 秒有啊」→ 169.93s（−72dB、190ms）✅

## 修后（apply_silfix2 → 本模块）
| | 修前 | 修后 | EL |
|---|---|---|---|
| p5 | −73.5 | **−59.1** | −60.5 |
| p10 | −62.4 | **−57.9** | −58.7 |
| 中位 | −51.9 | −51.3 | −51.8 |
| min | −240 | −86.5 | −98.5 |
| <−100dB 帧 | 32 | **0** | 0 |
| 深谷 | 20 段 | 0 段 | 8 段 |
| 点名处 | −73/−240/−72 | **−59/−59/−60** | — |
| L1 Σz² | 32.2 | **27.8** | — |
| L1 平坦度 | +3.85σ | **+3.22σ** | — |
L1 闸门（probe_g）两项都**变好**，发声占比不变，整体 RMS 只掉 0.5dB。

## 三条硬规矩（写死，别再犯）
1. **加法式地板**：`out = seg + tone*g`。只会抬高，绝不压低 → 物理上不可能挖出新坑。
   🔴 严禁 `where(|tone| > |seg|, tone, seg)` 这种「逐样本取较大」写法 ——
   噪声过零时 |tone| 会掉到 |seg| 以下，数字零原样漏过去。
   app/declick.py 里那个漏掉 32 帧数字零的漏洞就是同一款。
2. **房间底噪自造**：优先取自身落在目标带内的帧拼接（频谱自洽）；不够再放宽；
   最后才用 1/f 粉噪，且**必须按 RMS 归一**到目标电平（峰值归一 = 电平差 16dB，
   旧 apply_silfix 就是这么把 −52 变成 −68 的）。
3. **只认真正的停顿**：段长 ≥60ms，且比前后各 300ms 上下文低 >12dB。
   否则会往轻读音节上加噪声，变成新的「顶起」。

## 前九次失败（都别重踩）
① 软噪声门压拖尾 ② 填房间底噪(旧) ③ 句末淡出 ④ 元音起音柔化 ⑤ 分段压顶起
⑥ 迭代压顶起 ⑦ 连续平滑增益 ⑧ 句末突起压制 ⑨ 填静音(旧版三处 bug)

## 位置
必须在 declick **之后**、libmp3lame **之前**（与 declick 同理：NEW_ENGINES 走的是
_legacy_post=False 的快路径，插在 roomtone 后面会整条被绕过）。
"""
import os

import numpy as np

HOP_S = 0.005
FLOOR_DB = float(os.environ.get("TTS_SILFLOOR_DB", "-52"))
GATE_DB = float(os.environ.get("TTS_SILFLOOR_GATE", "-58"))
MINLEN_MS = float(os.environ.get("TTS_SILFLOOR_MINLEN", "60"))
CTX_DB = float(os.environ.get("TTS_SILFLOOR_CTX", "12"))
DIL_MS = float(os.environ.get("TTS_SILFLOOR_DIL", "20"))

ENABLED = os.environ.get("TTS_SILFLOOR", "1").strip().lower() not in (
    "0", "false", "no", "off")


def _env_db(y, sr, hop=None):
    hop = hop or max(1, int(HOP_S * sr))
    n = len(y) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float64), hop
    m = y[:n * hop].astype(np.float64).reshape(n, hop)
    e = np.sqrt((m ** 2).mean(axis=1))
    return 20.0 * np.log10(e + 1e-12), hop


def _runs(mask):
    if mask.size == 0:
        return []
    d = np.diff(mask.astype(np.int8))
    st = list(np.where(d == 1)[0] + 1)
    en = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        st.insert(0, 0)
    if mask[-1]:
        en.append(len(mask))
    k = min(len(st), len(en))
    return list(zip(st[:k], en[:k]))


def _pink(n, rng):
    """1/f 粉噪（兜底用）"""
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


def _room_tone(y, db, sr, floor_db, hop, rng):
    """优先取自身音频里落在目标带内的帧拼接成底噪；不够才用粉噪（按 RMS 归一）。"""
    fh = hop
    for lo, hi in ((floor_db - 4, floor_db + 6), (floor_db - 8, floor_db + 8),
                   (floor_db - 12, floor_db + 10)):
        idx = np.where((db > lo) & (db < hi))[0]
        if len(idx) * HOP_S < 0.3:
            continue
        parts = [y[i * fh:(i + 1) * fh] for i in idx[:4000]]
        return np.concatenate(parts).astype(np.float32), "own"
    n = max(int(sr * 1.0), 8)
    return _pink(n, rng).astype(np.float32), "pink"


def apply(y, sr=24000, floor_db=None, gate_db=None, minlen_ms=None,
          ctx_db=None, dil_ms=None, enabled=None):
    """抬起句间停顿地板。**不改长度**。返回新数组（float32）。"""
    if enabled is None:
        enabled = ENABLED
    if not enabled or y is None or len(y) < int(sr * 0.5):
        return y
    floor_db = FLOOR_DB if floor_db is None else floor_db
    gate_db = GATE_DB if gate_db is None else gate_db
    minlen_ms = MINLEN_MS if minlen_ms is None else minlen_ms
    ctx_db = CTX_DB if ctx_db is None else ctx_db
    dil_ms = DIL_MS if dil_ms is None else dil_ms

    out = np.asarray(y, dtype=np.float32).copy()
    db, hop = _env_db(out, sr)
    if db.size == 0:
        return out
    rng = np.random.default_rng(20261016)
    amp = 10.0 ** (floor_db / 20.0)
    tone, src = _room_tone(out, db, sr, floor_db, hop, rng)
    tone = tone - tone.mean()
    tone = tone * (amp / (np.sqrt((tone ** 2).mean()) + 1e-20))

    cand = [(s, e) for s, e in _runs(db < gate_db)
            if (e - s) * HOP_S * 1000.0 >= minlen_ms]
    edge = set((s, e) for s, e in _runs(db < -90)
               if s == 0 or e >= len(db) - 1)          # 首尾补零段
    cand = sorted(set(cand) | edge)
    if not cand:
        return out
    ctx_n = max(1, int(0.30 / HOP_S))
    d_n = max(0, int(dil_ms / 1000.0 / HOP_S))
    fw = max(1, int(dil_ms / 1000.0 * sr))
    n_done = 0
    for s, e in cand:
        if (s, e) in edge:
            s2, e2 = s, e
        else:
            c0, c1 = max(0, s - ctx_n), min(len(db), e + ctx_n)
            ctx = np.concatenate([db[c0:s], db[e:c1]]) if c1 > c0 else None
            if ctx is not None and len(ctx) and \
                    float(np.median(ctx)) - float(np.median(db[s:e])) < ctx_db:
                continue                        # 上下文没高出一截 → 不是停顿
            s2, e2 = max(0, s - d_n), min(len(db), e + d_n)
        a, b = s2 * hop, e2 * hop
        n = b - a
        if n <= 0:
            continue
        seg = out[a:b]
        off = int(rng.integers(0, max(1, tone.size - n)))
        nz = np.tile(tone, n // max(1, tone.size) + 2)[off:off + n]
        g = np.ones(n, dtype=np.float32)
        if n > 2 * fw:
            g[:fw] = np.linspace(0.0, 1.0, fw)
            g[-fw:] = np.linspace(1.0, 0.0, fw)
        out[a:b] = seg + nz * g                  # 🔴 加法：只会抬高
        n_done += 1
    return out


def stats(y, sr=24000):
    """调试用：返回 (深谷段数, <−100dB 帧数, 停顿 p5, 停顿中位)。"""
    db, _ = _env_db(y, sr)
    if db.size == 0:
        return 0, 0, 0.0, 0.0
    lo = db[db < -45]
    p5 = float(np.percentile(lo, 5)) if lo.size else 0.0
    med = float(np.median(lo)) if lo.size else 0.0
    vs = [(s, e) for s, e in _runs(db < -60) if (e - s) * HOP_S >= 0.04]
    return len(vs), int((db < -100).sum()), p5, med
