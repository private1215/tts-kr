# -*- coding: utf-8 -*-
"""停顿预算对齐（Pause Budget Alignment, PBA）—— 把「断句碎」修掉。

## 为什么需要它

案例原声（ElevenLabs）的停顿率比我们低 36%（男）/ 81%（女），
生成的语音听起来「一顿一顿」。2026-09-30 全篇（男 1999 字 / 女 1773 字）
实测：

    指标        案例男声   我们 r3   案例女声   我们 r3
    句数        70        96        23        42
    句均时长    3.32s     2.34s     10.6s     5.6s
    停顿率/s    0.272     0.369     0.095     0.172
    静音占比    11.2%     13.8%     3.0%      5.8%

## 为什么不能在生成侧修（都已实测排除，别再试）

1. **换参考音频（含"挑最连贯的呼吸群"）无效且反向**。
   r2 用的 `EL_female_dyn` 是 14.64s 里只有 1 个停顿的最连续片段，
   结果输出 67 句（案例 23 句），是所有轮次里最碎的。
   两个性别同向反证：参考越连贯 → 输出越碎（男 0.433→0.325 / 0.323→0.365）。
2. **不是文本前端切碎的**：`cosy_server.py` 走 `text_frontend=False`
   （且 ttsfrd 未安装），1999 字只切成 7 块，而 VAD 检出 95 句。
3. **不是分块太多**：`CHUNK_MAX_CHARS=2600`，1999 字只有 1 块。
4. **调 `max_chars` 无效**：服务端 clamp，`--mc 600` 与默认逐字节一致。
5. 根因是 **AR 模型涌现停顿**（CosyVoice #1079/#1176、Flamed-TTS
   arXiv 2510.02848）：停顿位置由标点决定、时长由采样决定，没有开关。

## 它做什么

只做**静音手术**，不进声码器、不改任何有声样本，所以没有音质损失：

1. 在 16 kHz 上做 VAD（口径与 `pros.py` / `pdiag.py` 完全一致，
   校准系数就是在这个口径下标定的，换了口径 α 就不再成立）
2. 每个停顿 × α（默认 0.8）；被压到 250 ms 以下的会被 VAD 自然合并
3. 丢掉的静音时间用 atempo 把整段**拉回原时长**——
   这一步同时把发音速率也拉慢了约 4%，正好对上案例原声
   （案例有声段 6.05 字/s，我们原来 6.25 字/s）

α=0.8 全量对比（案例原声 → r3 原样 → α=0.8）：

    指标        男 GT    男 r3    男 0.8   女 GT    女 r3    女 0.8
    f0_mean     183.4   184.4   183.7   284.0   290.7   290.5
    onset_f0    200.5   185.9   184.6   307.8   292.4   304.1
    sent_dyn    12.46   9.54    9.96    11.63   11.11   11.61
    sent_dur    3.32    2.34    3.29    10.6    5.6     10.08
    n_sent      70      96      72      23      42      24
    ap_density  4.07    3.25    3.67    4.32    3.71    3.81
    cps         5.369   5.39    5.39    5.093   5.131   5.131
"""
import logging

import numpy as np

log = logging.getLogger("tts")

SR_A = 16000        # 分析采样率（校准口径，勿改）
GAP = 0.25          # 静音 ≥250ms 视为句边界（与 pros.py 一致）
MIN_SENT = 0.25     # 短于 250ms 的有声碎段丢弃
FLOOR_MS = 60       # 压缩后保留的最短静音，避免两个词直接贴死


def _to16(y, sr):
    """降到 16 kHz 做分析。"""
    y = np.asarray(y, dtype=np.float32).flatten()
    if sr == SR_A:
        return y
    import librosa
    return librosa.resample(y, orig_sr=sr, target_sr=SR_A)


def _intervals(y16):
    """16 kHz 上的有声区间（口径与 pros.py 的 sentences() 完全一致）。"""
    import librosa
    iv = librosa.effects.split(y16, top_db=32, frame_length=1024,
                               hop_length=256)
    out, cur = [], None
    for s, e in iv:
        if cur and (s - cur[1]) / SR_A < GAP:
            cur[1] = e
        else:
            if cur:
                out.append(cur)
            cur = [s, e]
    if cur:
        out.append(cur)
    return [(s, e) for s, e in out if (e - s) / SR_A >= MIN_SENT]


def align(y, sr, alpha=0.8):
    """压缩停顿。返回 (新音频, restore_ratio)。

    restore_ratio = 新时长 / 原时长（< 1），调用方把它乘进 atempo 即可把
    总时长拉回原值；返回 1.0 表示没改动。
    """
    y = np.asarray(y)
    if alpha >= 0.999 or y.size < SR_A:
        return y, 1.0
    try:
        y16 = _to16(y, sr)
        ss = _intervals(y16)
    except Exception as e:
        log.warning("PBA: VAD 失败，跳过（%s）", repr(e)[:120])
        return y, 1.0
    if len(ss) < 2:
        return y, 1.0
    # 16k 上的样本下标 → 原始采样率
    k = float(sr) / SR_A
    ss_o = [(int(round(s * k)), int(round(e * k))) for s, e in ss]
    ps_o = [(ss_o[i][1], ss_o[i + 1][0]) for i in range(len(ss_o) - 1)]
    ps_o = [(a, b) for a, b in ps_o if b > a]
    if not ps_o:
        return y, 1.0

    floor = int(FLOOR_MS / 1000.0 * sr)
    pieces = [y[ss_o[0][0]:ss_o[0][1]]]
    for i in range(1, len(ss_o)):
        gs, ge = ps_o[i - 1]
        g = ge - gs
        pieces.append(y[gs:gs + max(floor, int(g * alpha))])
        pieces.append(y[ss_o[i][0]:ss_o[i][1]])
    y2 = np.concatenate(pieces)
    ratio = len(y2) / float(len(y))
    log.info("PBA alpha=%.2f: 停顿 %d 处，时长 %.2fs -> %.2fs (ratio %.4f)",
             alpha, len(ps_o), len(y) / float(sr), len(y2) / float(sr), ratio)
    return y2, ratio
