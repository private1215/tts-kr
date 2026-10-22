#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""head7_cmp53.py —— #7 型起音崩坏：到底能不能安全切除？

## 问题
headfix 现在只切「碎片组恰好 1 组」的 A 型。#7 型（起音抛多组碎片）占产出 9%，
保守不切。不切的原因：**短首句**（如「네. 반갑습니다…」）本身就会是多音节 +
句间停顿 ≥250ms，会被整段吃掉。

## 判别思路（本轮的做法）
不看时长、不看 dB —— **让 ASR 直接听头部区域**：
- 听出来是垃圾/幻觉 → 切除安全
- 听出来是文案首句的真词 → 不能切

这是唯一不依赖阈值拍脑袋的证据：ASR 听到的东西就是员工会听到的东西。
再配一个客观护栏：切掉的部分总时长 vs 首句在全文里的占比。

## 用法
venv/bin/python head7_cmp53.py [目录名...]   不给就扫最近 34 条
"""
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/tools")
sys.path.insert(0, "/mnt/tts/diag")          # headtype53 在 diag 不在 tools
from headtype53 import head_shape  # noqa: E402
from asr_check import normalize, transcribe  # noqa: E402

OUT = "/mnt/tts/output"


def head_region(y, sr, head_end):
    """头部区域 = [0, head_end)，按 head_shape 的算法。"""
    return y[:int(head_end * sr)]


def groups_of(y, sr, thr=-30.0):
    """头部区域里 >thr dB 的组（与 head_shape 同口径：10ms 帧 RMS）。"""
    n = int(0.01 * sr)
    nfr = len(y) // n
    if nfr == 0:
        return []
    fr = y[:nfr * n].reshape(nfr, n)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-12)
    out, i = [], 0
    while i < nfr:
        if db[i] > thr:
            j = i
            while j < nfr and db[j] > thr:
                j += 1
            out.append((i * 0.01, j * 0.01))
            i = j
        else:
            i += 1
    return out


def main():
    args = sys.argv[1:]
    if args:
        dirs = [os.path.join(OUT, a) for a in args]
    else:
        dirs = sorted(glob.glob(os.path.join(OUT, "*")),
                      key=os.path.getmtime, reverse=True)[:34]
    tmp = tempfile.mkdtemp(prefix="head7_")
    print("%-26s %-12s %6s %6s %5s %9s  %s"
          % ("目录", "形态", "头区长", "组数", "语音s", "ASR听到", "判定"))
    print("-" * 108)
    for d in dirs:
        m3 = sorted(glob.glob(os.path.join(d, "*.mp3")))
        if not m3:
            continue
        wav = os.path.join(tmp, os.path.basename(d) + ".wav")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y",
                        "-i", m3[0], "-ac", "1", "-ar", "16000", wav],
                       check=True, timeout=120)
        y, sr = sf.read(wav)
        kind, dur, gap, ng = head_shape(y, sr)
        if kind == "normal":
            continue
        # head_end：从 S 往前找第一个 >=150ms 深间隙（与 head_shape 同算法）
        n = int(0.01 * sr)
        nfr = len(y) // n
        fr = y[:nfr * n].reshape(nfr, n)
        db = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-12)
        v = db > -28.0
        # 找 S：首个 >-25dB 且其后 300ms 均值 >-28dB
        S = None
        for i in range(nfr):
            if db[i] > -25 and v[max(0, i):i + 30].mean() > 0.5:
                S = i * 0.01
                break
        if S is None:
            print("%-26s %-12s  (找不到起音点)" % (os.path.basename(d)[:24], kind))
            continue
        # 从 S 往前跳过 lead-in（<150ms 浅间隙），找第一个 >=150ms 深间隙
        he = S
        i = int(S / 0.01)
        while i > 0:
            j = i
            while j > 0 and db[j - 1] <= -28:
                j -= 1
            gapd = (i - j) * 0.01
            if gapd >= 0.15:
                he = j * 0.01
                break
            i = j
        hy = head_region(y, sr, he)
        gs = groups_of(hy, sr)
        speech = sum(b - a for a, b in gs)
        # ASR 听头部（头部太短(<0.3s)就没有意义，跳过）
        if he < 0.30:
            txt = "(头区<0.3s)"
            verdict = "—"
        else:
            hwav = os.path.join(tmp, os.path.basename(d) + "_head.wav")
            sf.write(hwav, hy, sr)
            try:
                txt = normalize(transcribe(hwav, language="Korean"))
            except Exception as e:
                txt = "(ASR失败 %r)" % repr(e)[:40]
            verdict = "🔴垃圾/幻觉" if not txt else "⚠️有词(需核对文案)"
        print("%-26s %-12s %6.2f %6d %5.2f %9s  %s"
              % (os.path.basename(d)[:24], kind, he, len(gs), speech,
                 txt[:14] or "(空)", verdict))
        print("      组: %s" % " ".join("%.2f-%.2f(%.2fs)" % (a, b, b - a)
                                        for a, b in gs[:8]))
        os.remove(wav)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
