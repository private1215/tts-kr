# -*- coding: utf-8 -*-
"""q5_headfix.py v3 —— 文件头孤立段修复器

v2 的链条扫描把 S 之前的主语音起音段误当成碎片（#2 chain=[(0.78,0.87)]
其实是持续语音本身）。v3 重写为两段式：
  1) 从 S 往前跳过 lead-in（语音 + <150ms 浅间隙），找到第一个深度静音
     （>=150ms 且 min < -47dB），其终点 = 切点
  2) 头部区域 [0, 断口起点) 内按深度静音分「碎片组」，判据：
     - 恰好 1 组（#4 词内断裂有 2 组 -> 不切）
     - 组总时长 <= 350ms
     - 组起点 < 0.30s（紧贴文件头）
     - 切点后 400ms 内恢复持续正常语音（安全闸）

实测（headrepro-1019 十条）：
  #2/#3/#10 = 起音即碎片 + 深静音 + 真语音 -> CUT（ASR 证明无损）
  #4 = 「로보티즈」被 300ms 断口撕成 2 组 -> keep（切了吃字）
  其余 = 正常起音 -> keep
"""
import os
import sys

import numpy as np
import soundfile as sf

SR = 16000
DEEP_MIN = -47.0    # 深度静音阈值
GAP_MS = 150        # 断口最小时长 ms
FRAG_DB = -30.0     # 碎片判定电平
FRAG_MAX = 0.35     # 碎片组总时长上限 s
FRAG_HEAD = 0.30    # 碎片组起点距文件头上限 s


def envdb(x, sr):
    hop = max(1, int(0.01 * sr))
    n = len(x) // hop
    return 20 * np.log10(np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-12)


def speech_onset(v):
    """首个 >-25dB 且其后 300ms 均值 >-28dB 的位置（单位 s）"""
    w = 30  # 300ms @10ms/帧
    for i in range(len(v) - w):
        if v[i] > -25 and v[i:i + w].mean() > -28:
            return i * 0.01
    return None


def head_groups(v, head_end):
    """头部区域 [0, head_end) 内按深度静音分碎片组，返回 [(s,e),...]（单位 s）"""
    groups = []
    k = 0
    while k < head_end:
        if v[k] > FRAG_DB:
            m = k
            while m < head_end and v[m] > FRAG_DB:
                m += 1
            if groups and (k - groups[-1][1]) * 0.01 < GAP_MS / 1000.0:
                pass  # 与上一组间隔太短，理论上不会发生（组间必为深度静音）
            groups.append((k * 0.01, m * 0.01))
            k = m
        else:
            m = k
            while m < head_end and v[m] <= FRAG_DB:
                m += 1
            k = m
    return groups


def find_cut(v, verbose=False):
    """返回 (切点秒 or None, 诊断dict)"""
    S = speech_onset(v)
    if S is None:
        return None, {"reason": "no_speech"}
    si = int(round(S / 0.01))
    # 1) 从 S 往前跳过 lead-in，找第一个深度静音断口
    i = si - 1
    lead = []
    while i >= 0:
        if v[i] > FRAG_DB:
            j = i
            while j >= 0 and v[j] > FRAG_DB:
                j -= 1
            lead.append((round((j + 1) * 0.01, 2), round((i + 1) * 0.01, 2)))
            i = j
        else:
            j = i
            while j >= 0 and v[j] <= FRAG_DB:
                j -= 1
            gap = (i - j) * 0.01
            seg_min = float(v[j + 1:i + 1].min()) if i > j else 0.0
            if gap >= GAP_MS / 1000.0 and seg_min < DEEP_MIN:
                break
            if gap >= GAP_MS / 1000.0:
                break  # 长但不够深的间隙也断（保守）
            i = j
    d = {"S": round(S, 2), "lead": lead}
    if i < 0:
        return None, dict(d, reason="no_deep_gap")
    cut = (i + 1) * 0.01
    d["cut"] = round(cut, 2)
    # 2) 头部区域碎片分组
    groups = head_groups(v, i + 1)
    d["groups"] = [(round(a, 2), round(b, 2)) for a, b in groups]
    if len(groups) != 1:
        return None, dict(d, reason="multi_group" if groups else "no_fragment")
    dur = groups[0][1] - groups[0][0]
    d["frag_dur"] = round(dur, 3)
    if dur > FRAG_MAX:
        return None, dict(d, reason="frag_too_long")
    if groups[0][0] > FRAG_HEAD:
        return None, dict(d, reason="frag_not_at_head")
    # 3) 安全闸：切点后 400ms 内恢复持续正常语音
    k_end = min(len(v), i + 1 + 40)
    if not (i + 1 < len(v) and v[i + 1:k_end].max() > -25):
        return None, dict(d, reason="no_speech_after_gap")
    return cut, dict(d, reason="cut")


def fix_head(x, sr=SR, fade_ms=5):
    v = envdb(x, sr)
    cut, d = find_cut(v)
    if cut is None:
        return x, d
    n = int(round(cut * sr))
    y = x[n:].copy()
    fade = min(len(y), int(fade_ms * sr / 1000))
    if fade > 0:
        y[:fade] *= np.linspace(0, 1, fade)
    return y, d


def click_count(x):
    x = x / (np.abs(x).max() + 1e-12) * 0.70
    d = np.abs(np.diff(x))
    idx = np.where(d >= 0.30)[0]
    if len(idx) == 0:
        return 0
    n = 1
    for a, b in zip(idx, idx[1:]):
        if b - a > 1:
            n += 1
    return n


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "headrepro":
        files = ["/mnt/tts/diag/v0902/headrepro-1019-%d.16k.wav" % k
                 for k in range(1, 11)]
    else:
        files = args
    for p in files:
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        y, d = fix_head(x, sr)
        tag = os.path.basename(p)
        if d.get("reason") == "cut":
            print("%-30s CUT  cut=%.2f S=%s lead=%s groups=%s dur=%.3f click %d->%d"
                  % (tag, d["cut"], d["S"], d["lead"], d["groups"],
                     d.get("frag_dur", -1), click_count(x), click_count(y)))
        else:
            print("%-30s keep reason=%-18s S=%s lead=%s groups=%s"
                  % (tag, d.get("reason"), d.get("S"), d.get("lead"),
                     d.get("groups")))
