# -*- coding: utf-8 -*-
"""参考音频自动裁剪 + 文字稿自动对齐。

设计目标：员工只需要「上传整段录音 + 粘贴整段文字稿」，不需要懂剪辑、
不需要自己数秒数。服务端挑出音质最好的一段，并把文字稿按**实际说话时间**
（而不是墙钟时间）对齐截出对应片段。

为什么不能按墙钟时间切文字：录音里有停顿、有语速快慢，15 秒的音频未必
对应「总字数 × 15/总时长」那么多字。按累计**有声时长**做单调映射，
能在语速不均时明显贴近真实位置。

为什么不能只按累计有声时间硬切：那样会切在半个句子甚至半个词上。
所以先按句子边界吸附，吸附结果偏离太远才退回按词边界切。
"""
import logging
import re

import numpy as np

log = logging.getLogger("tts")

HOP_SEC = 0.02            # 短时能量帧长
MIN_PAUSE_SEC = 0.22      # 视为「可切开的停顿」的最短静音
LONG_PAUSE_SEC = 0.80     # 窗口内部出现这么长的停顿要扣分
TARGET_SEC = 15.0         # 期望截取长度
MIN_CLIP_SEC = 8.0
MAX_CLIP_SEC = 20.0
SNAP_SEC = 0.60           # 边界在 ±0.6s 内吸附到能量最低点（避免切在字中间）
FADE_SEC = 0.02

# 句子分隔符：韩/中/日/英通用
_SENT_END = ".!?。！？…\n"


def short_time_rms(x: np.ndarray, sr: int):
    """返回 (rms 数组, 帧长秒)。"""
    hop = max(1, int(sr * HOP_SEC))
    nf = len(x) // hop
    frames = x[:nf * hop].reshape(nf, hop)
    return np.sqrt((frames ** 2).mean(axis=1) + 1e-12), hop


def _speech_mask(rms: np.ndarray) -> np.ndarray:
    """自适应阈值的有声/静音判定。

    用低分位当噪底、中分位当语音参考，避免固定阈值在不同录音电平下失效。
    """
    if rms.size == 0:
        return np.zeros(0, dtype=bool)
    floor = np.percentile(rms, 10)
    mid = np.percentile(rms, 60)
    thr = max(floor * 2.5, mid * 0.35, 1e-4)
    return rms > thr


def _pooled(rms: np.ndarray, k: int) -> np.ndarray:
    """把帧级 rms 按 k 帧合并（用于算每秒均值/最小值）。"""
    n = (len(rms) // k) * k
    return rms[:n].reshape(-1, k).mean(axis=1)


def _snap(idx: int, rms: np.ndarray, half: int) -> int:
    """在 idx ± half 帧内找平滑能量最低点，避免切在字中间。"""
    lo = max(0, idx - half)
    hi = min(len(rms), idx + half)
    if hi <= lo:
        return idx
    k = 25  # 0.5s 平滑
    sm = np.convolve(rms, np.ones(k) / k, mode="same")
    return lo + int(np.argmin(sm[lo:hi]))


def pick_window(x: np.ndarray, sr: int, target: float = TARGET_SEC):
    """挑出音质最好的一段。返回 (start_sec, end_sec, info)。

    打分 = 平均能量 × 稳定度 × 停顿惩罚：
      - 平均能量高：信噪比好，克隆更干净
      - 稳定度高（cv 小）：音量平稳，不会有忽远忽近
      - 停顿惩罚：窗口内部的长停顿会打断语流，扣分
    """
    dur = len(x) / float(sr)
    rms, hop = short_time_rms(x, sr)
    n = len(rms)
    target_f = int(target / HOP_SEC)

    if dur <= target:
        return 0.0, dur, {"method": "no_trim", "dur": round(dur, 2)}

    # 有停顿的位置才是干净的切点；连续语音（没有长停顿）时退化为逐帧滑窗
    mask = _speech_mask(rms)
    pauses = []
    i = 0
    while i < n:
        if not mask[i]:
            j = i
            while j < n and not mask[j]:
                j += 1
            if (j - i) * HOP_SEC >= MIN_PAUSE_SEC:
                pauses.append((i + j) // 2)
            i = j
        else:
            i += 1

    cum = np.concatenate([[0.0], np.cumsum(rms)])
    cum2 = np.concatenate([[0.0], np.cumsum(rms ** 2)])
    per_sec = max(1, int(1.0 / HOP_SEC))
    pooled = _pooled(rms, per_sec)

    def score(sf, ef):
        m = (cum[ef] - cum[sf]) / (ef - sf)
        v = max(0.0, (cum2[ef] - cum2[sf]) / (ef - sf) - m * m)
        cv = np.sqrt(v) / m if m > 0 else 9.9
        stab = 1.0 / (1.0 + min(cv, 3.0))
        # 内部长停顿
        seg = mask[sf:ef]
        worst = 0
        run = 0
        for b in seg:
            if not b:
                run += 1
                worst = max(worst, run)
            else:
                run = 0
        pen = 1.0 / (1.0 + max(0.0, worst * HOP_SEC - LONG_PAUSE_SEC) * 2.0)
        # 内部最低 1 秒能量（不能有一秒几乎没声）
        ps = pooled[sf // per_sec: max(sf // per_sec + 1, ef // per_sec)]
        mn = ps.min() if ps.size else 0.0
        if mn < 0.02:
            pen *= 0.35
        return m * stab * pen, m, cv

    best = None
    if len(pauses) >= 2:
        # 优先用停顿点做边界：切在句间，最干净
        bounds = [0] + pauses + [n]
        for pi, sf in enumerate(bounds):
            for ef in bounds[pi + 1:]:
                L = (ef - sf) * HOP_SEC
                if L < MIN_CLIP_SEC or L > MAX_CLIP_SEC:
                    continue
                s, m, cv = score(sf, ef)
                if best is None or s > best[0]:
                    best = (s, sf, ef, m, cv, "pause")
    if best is None:
        # 连续语音：滑窗 + 边界吸附到能量最低点
        step = max(1, int(0.5 / HOP_SEC))
        for sf in range(0, max(1, n - target_f), step):
            ef = min(n, sf + target_f)
            if (ef - sf) * HOP_SEC < MIN_CLIP_SEC:
                continue
            s, m, cv = score(sf, ef)
            if best is None or s > best[0]:
                best = (s, sf, ef, m, cv, "slide")

    if best is None:
        return 0.0, min(dur, MAX_CLIP_SEC), {"method": "fallback_head"}

    _, sf, ef, m, cv, method = best
    half = int(SNAP_SEC / HOP_SEC)
    sf2 = _snap(sf, rms, half)
    ef2 = _snap(ef, rms, half)
    if (ef2 - sf2) * HOP_SEC < MIN_CLIP_SEC:
        sf2, ef2 = sf, ef
    a = sf2 * HOP_SEC
    b = min(dur, ef2 * HOP_SEC)
    return a, b, {
        "method": method,
        "mean_rms": round(float(m), 4),
        "cv": round(float(cv), 3),
        "raw_start": round(sf * HOP_SEC, 2),
        "raw_end": round(ef * HOP_SEC, 2),
        "pauses": len(pauses),
    }


def _sentences(text: str):
    """按句末标点切分，返回 [(start, end), ...]（含分隔符本身）。"""
    out = []
    start = 0
    for i, ch in enumerate(text):
        if ch in _SENT_END:
            # 吃掉分隔符后连续的同类符号与换行
            j = i + 1
            while j < len(text) and text[j] in _SENT_END:
                j += 1
            out.append((start, j))
            start = j
    if start < len(text):
        out.append((start, len(text)))
    return [(s, e) for s, e in out if e > s and text[s:e].strip()]


def _word_snap(text: str, i0: int, i1: int, limit: int = 24):
    """把字符区间吸附到词/字边界：起点往前找空白，终点往后找空白。"""
    n = len(text)
    a = max(0, min(i0, n))
    k = 0
    while a > 0 and text[a - 1] not in " \t\n" and k < limit:
        a -= 1
        k += 1
    b = max(a, min(i1, n))
    k = 0
    while b < n and text[b] not in " \t\n" and k < limit:
        b += 1
        k += 1
    return a, b


def slice_transcript(text: str, mask: np.ndarray, hop: int, a: float, b: float):
    """按累计有声时间，把整段文字稿映射到 [a, b] 这段时间对应的片段。"""
    text = (text or "").strip()
    if not text:
        return "", {"method": "empty"}
    total_speech = float(mask.sum()) * hop
    total_dur = float(len(mask)) * hop
    ia = int(a / hop)
    ib = int(b / hop)
    if total_speech > 1e-6:
        fa = float(mask[:ia].sum()) * hop / total_speech
        fb = float(mask[:ib].sum()) * hop / total_speech
        basis = "speech_time"
    else:
        fa = a / max(total_dur, 1e-6)
        fb = b / max(total_dur, 1e-6)
        basis = "wall_clock"

    n = len(text)
    ca = max(0.0, min(1.0, fa)) * n
    cb = max(0.0, min(1.0, fb)) * n
    want = max(1.0, cb - ca)

    sents = _sentences(text)
    got = None
    if sents:
        s0 = None
        for s, e in sents:
            if s <= ca:
                s0 = s
            else:
                break
        e0 = None
        for s, e in sents:
            if e >= cb:
                e0 = e
                break
        if s0 is None:
            s0 = sents[0][0]
        if e0 is None:
            e0 = sents[-1][1]
        if e0 > s0:
            cand = text[s0:e0].strip()
            # 吸附后长度离谱（比如整篇没有标点，一句话吞掉全部）就弃用
            if len(cand) <= want * 2.2:
                got = (s0, e0, cand, "sentence")

    if got is None:
        i0, i1 = _word_snap(text, int(ca), int(cb))
        got = (i0, i1, text[i0:i1].strip(), "word")

    s0, e0, sub, method = got
    return sub, {
        "method": method,
        "basis": basis,
        "char_range": (s0, e0),
        "want_chars": int(round(want)),
        "got_chars": len(sub),
        "total_chars": n,
        "speech_ratio": round(total_speech / max(total_dur, 1e-6), 3),
    }


def trim_reference(x: np.ndarray, sr: int, ref_text: str,
                   target: float = TARGET_SEC, trim_above: float = 30.0):
    """完整流程：挑窗口 -> 裁音频 -> 对齐裁文字。

    返回 (audio, sr, ref_text, info)。音频长度未超过 trim_above 时原样返回。
    """
    x = np.asarray(x, dtype=np.float32).flatten()
    dur = len(x) / float(sr)
    info = {"src_dur": round(dur, 2), "trimmed": False}
    if dur <= trim_above:
        info["method"] = "no_trim"
        return x, sr, ref_text, info

    a, b, winfo = pick_window(x, sr, target=target)
    info.update(winfo)
    info["trimmed"] = True
    info["clip_start"] = round(a, 2)
    info["clip_end"] = round(b, 2)
    info["clip_dur"] = round(b - a, 2)

    i0, i1 = int(a * sr), int(b * sr)
    seg = x[i0:i1].copy()
    fade = max(1, int(sr * FADE_SEC))
    if seg.size > 2 * fade:
        seg[:fade] *= np.linspace(0.0, 1.0, fade).astype(np.float32)
        seg[-fade:] *= np.linspace(1.0, 0.0, fade).astype(np.float32)

    rms, hop = short_time_rms(x, sr)
    mask = _speech_mask(rms)
    sub, tinfo = slice_transcript(ref_text, mask, HOP_SEC, a, b)

    # 兜底：对齐结果太短（文字稿与音频对不上、或整篇没标点被整段吞掉）时，
    # 退回按墙钟时间比例切。ref_text 过短会让模型提前收尾，表现为「只念了一半」。
    want = (b - a) * 6.15
    if len(sub) < max(10, 0.35 * want) and len(ref_text.strip()) > len(sub):
        n = len(ref_text.strip())
        i0, i1 = _word_snap(ref_text.strip(),
                            int(a / max(dur, 1e-6) * n),
                            int(b / max(dur, 1e-6) * n))
        cand = ref_text.strip()[i0:i1].strip()
        if len(cand) > len(sub):
            sub = cand
            tinfo["method"] = tinfo.get("method", "") + "+fallback_wallclock"
    info.update({"text_" + k: v for k, v in tinfo.items()})
    info["want_chars"] = int(round(want))
    log.info("参考音频自动截取: %.2fs -> [%.2f, %.2f] %.2fs | %s | 文字 %s",
             dur, a, b, b - a, winfo.get("method"), tinfo)
    return seg, sr, sub, info
