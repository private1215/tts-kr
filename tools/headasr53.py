#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""headasr53.py —— #7 型头部内容判定：头部区域里到底有没有真词？

## 背景
shorthead53 实验结论：短首句「반갑습니다. 오늘도 함께하겠습니다.」的头型是
#7_collapse（3 组、0.51s 语音），ASR 听到真词「반갑습니다」→ 按形状切 #7 会吃真词。
但那是「短首句」这一类。真实 #7（로보티즈 开头，占产出 9%）的头部是不是也是真词？
→ 对两边都做「头部 ASR + 全篇 ASR」，看真词在不在头部。

## 判定
- 头部 ASR ≈ 文案开头 → 头部是真词 → 不能切
- 头部 ASR 空/垃圾/幻觉 → 头部是垃圾 → 可切（还需形状判据配合）

用法：venv/bin/python tools/headasr53.py
"""
import glob
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")
sys.path.insert(0, "/mnt/tts/diag")
from asr_check import normalize, transcribe  # noqa: E402
from headtype53 import head_shape  # noqa: E402

OUT = "/mnt/tts/output"

# (目录, 文案)
SAMPLES = [
    ("frag-P_robo-s42", "로보티즈 주주 여러분 반갑습니다 오늘도 좋은 하루 보내세요"),
    ("frag-P_robo-s1001", "로보티즈 주주 여러분 반갑습니다. 오늘도 좋은 하루 보내세요."),
    ("frag-P_l3f-s42", "반갑습니다 암호텍 주주 여러분 이민정입니다"),
    ("frag-P_l3f-s1001", "반갑습니다. 아모텍 주주 여러분 이민정입니다."),
    ("frag-P_han-s42", "한 퍼센트 정도 올랐습니다. 전일대비 상승폭이 컸습니다."),
    ("frag-P_han-s1001", "한 퍼센트 정도 올랐습니다. 전일 대비 상승폭이 컸습니다."),
    ("sh53_네+반갑_EL_female", "네. 반갑습니다."),
    ("sh53_안녕+반갑_EL_female", "안녕하세요. 반갑습니다."),
    ("sh53_네+안녕_EL_female", "네. 안녕하세요."),
    ("sh53_반갑+오늘_EL_female", "반갑습니다. 오늘도 함께하겠습니다."),
    ("sh53_오늘은 삼성_EL_female", "오늘은 삼성전자에 대해 말씀드리겠습니다."),
]


def to16k(mp3, wav):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", mp3,
                    "-ac", "1", "-ar", "16000", wav], check=True)
    return sf.read(wav)


def head_groups(y, sr, he):
    """头部区域 [0, he) 里的 >-30dB 组：[(起点, 时长), ...]"""
    hop = int(0.01 * sr)
    n = min(len(y) // hop, 300)
    db = 20 * np.log10(np.array(
        [np.sqrt(np.mean(y[i * hop:(i + 1) * hop] ** 2) + 1e-12)
         for i in range(n)]) + 1e-12)
    vv = db > -30.0
    hi = min(int(he * 100), n)
    gs, k = [], 0
    while k < hi:
        if vv[k]:
            m = k
            while m < hi and vv[m]:
                m += 1
            gs.append((k * 0.01, (m - k) * 0.01)); k = m
        else:
            k += 1
    return gs


def head_end_of(y, sr):
    """头部区域右界（与 head_shape 同算法：3s 窗、-30dB、>=150ms 深间隙）。
    返回 (S 持续语音起点秒, 头部终点秒)；头区=0 表示起音即正文。"""
    hop = int(0.01 * sr)
    n = min(len(y) // hop, 300)
    db = 20 * np.log10(np.array(
        [np.sqrt(np.mean(y[i * hop:(i + 1) * hop] ** 2) + 1e-12)
         for i in range(n)]) + 1e-12)
    v = db > -30.0
    si = None
    for k in range(len(db) - 30):
        if db[k] > -25 and db[k:k + 30].mean() > -28:
            si = k
            break
    if si is None:
        return None, None
    i = si - 1
    head_end = None
    while i >= 0:
        if v[i]:
            j = i
            while j >= 0 and v[j]:
                j -= 1
            i = j
        else:
            j = i
            while j >= 0 and not v[j]:
                j -= 1
            if (i - j) >= 15:
                head_end = j + 1
                break
            i = j
    if head_end is None or head_end <= 0:
        return si * 0.01, 0.0
    return si * 0.01, head_end * 0.01


def main():
    rows = []
    for tag, text in SAMPLES:
        # 🔴 产物目录会加 -N 后缀，必须按 mtime 取最新（不能 glob 排序末尾）
        dirs = sorted(glob.glob(os.path.join(OUT, tag + "*")),
                      key=os.path.getmtime)
        mp3 = None
        for d in reversed(dirs):
            fs = sorted(glob.glob(os.path.join(d, "*.mp3")))
            if fs:
                mp3 = fs[-1]
                break
        if not mp3:
            print("❌ %s 无产物" % tag); continue
        y, sr = to16k(mp3, "/tmp/ha53.wav")
        kind, fdur, fgap, ng = head_shape(y, sr)
        S, he = head_end_of(y, sr)
        gs = head_groups(y, sr, he or 0.0)
        # 头部区域 ASR
        sf.write("/tmp/ha53_head.wav", y[:int((he or 0) * sr)], sr)
        try:
            htxt = normalize(transcribe("/tmp/ha53_head.wav", language="Korean"))
        except Exception as e:
            htxt = "(失败 %r)" % repr(e)[:30]
        # 全篇 ASR
        try:
            ftxt = normalize(transcribe(mp3.replace(".mp3", ".wav")
                                        if os.path.exists(mp3.replace(".mp3", ".wav"))
                                        else "/tmp/ha53.wav", language="Korean"))
        except Exception as e:
            ftxt = "(失败 %r)" % repr(e)[:30]
        print("─" * 70)
        print("%s  文案=%r" % (tag, text))
        print("  kind=%-12s S=%.2f 头区=%.2fs 组=%s" % (
            kind, S if S is not None else -1, he or -1,
            " ".join("%.2f+%.2f" % g for g in gs[:8]) or "(空)"))
        print("  头部ASR=%r" % htxt[:60])
        print("  全篇ASR=%r" % ftxt[:80])
        rows.append({"tag": tag, "text": text, "kind": kind, "head": fdur,
                     "groups": gs, "head_asr": htxt, "full_asr": ftxt})
        sys.stdout.flush()
    import json
    json.dump(rows, open("/mnt/tts/diag/headasr53.json", "w"),
              ensure_ascii=False, indent=1)
    print("\n=== 汇总：头部 ASR 是否命中文案开头 ===")
    for r in rows:
        t = r["text"].replace(" ", "")
        h = r["head_asr"].replace(" ", "")
        hit = "✅真词" if h and (h in t or t.startswith(h[:2])) else (
            "🗑️ 垃圾/空" if not h else "❓待看")
        print("  %-24s kind=%-12s 头部ASR=%-14r %s" % (
            r["tag"], r["kind"], r["head_asr"][:14], hit))


if __name__ == "__main__":
    main()
