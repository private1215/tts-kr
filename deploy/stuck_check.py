#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核对员工反馈：「第二题 B 选项——卡字、吃字，中间几个字」

## 背景
z7 第二题 = 女声中段裁决（A=现役 / B=加 highshelf4000 −2dB）。
🔴 **A 和 B 是同一段音频**（都是女声成品 90~120s），只差一条 EQ 曲线。
   所以"卡字/吃字"如果真的存在，**两段都有**，EQ 不可能凭空造出字来。
   本脚本要回答两件事：
   1. 这段里到底有没有卡字/吃字（客观证据）
   2. EQ(B) 有没有让它变严重（A/B 逐项对比）

## 检测什么
- **卡字（stutter/重复）**：0.05~0.5s 短延迟上的音频自重复（同一小段被播了两遍）
- **吃字（吞音/截断）**：有声段在极短时间内（<15ms）掉到静音，且不是句末
- **硬切**：相邻 10ms 帧落差 > 60dB
- **异常短静音**（30~80ms，不是正常停顿，听着像"卡"）
"""
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

TMP = "/mnt/tts/diag/_stuck"
os.makedirs(TMP, exist_ok=True)

SRC = "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"
S0, DUR = 90.0, 30.0        # z7 第二题取的段
EQ = "highshelf=f=4000:g=-2"


def cut(af, tag):
    out = os.path.join(TMP, "%s.wav" % tag)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", SRC, "-ss", "%.1f" % S0,
           "-t", "%.1f" % DUR, "-ac", "1", "-ar", "16000"]
    if af:
        cmd += ["-af", af]
    cmd += [out]
    subprocess.run(cmd, check=True)
    y, sr = sf.read(out, dtype="float32")
    return y, sr


def stutter(y, sr):
    """找 0.05~0.5s 延迟上的高相似重复（卡字特征）。"""
    hop = int(0.02 * sr)
    win = int(0.10 * sr)
    n = (len(y) - win) // hop
    frames = np.array([y[i * hop:i * hop + win] for i in range(n)])
    frames /= (np.linalg.norm(frames, axis=1, keepdims=True) + 1e-9)
    hits = []
    for lag in range(3, 26):          # 60ms ~ 520ms
        if lag >= n:
            break
        sim = np.sum(frames[:-lag] * frames[lag:], axis=1)
        idx = np.where(sim > 0.97)[0]
        for i in idx:
            hits.append((i * hop / float(sr), lag * hop / float(sr), float(sim[i])))
    # 去重（相邻命中算一次）
    uniq = []
    for t, lg, s in sorted(hits):
        if uniq and t - uniq[-1][0] < 0.3:
            continue
        uniq.append((t, lg, s))
    return uniq


def cuts(y, sr):
    """硬切 + 突断 + 异常短静音。"""
    hop = int(0.01 * sr)
    n = len(y) // hop
    e = np.sqrt(np.array([(y[i * hop:(i + 1) * hop] ** 2).mean() for i in range(n)]))
    db = 20 * np.log10(e + 1e-12)
    d = np.diff(db)
    hard = np.where(np.abs(d) > 60)[0]
    thr = np.percentile(db, 30)
    sil = db < thr
    dd = np.diff(sil.astype(int))
    starts = np.where(dd == 1)[0] + 1
    ends = np.where(dd == -1)[0] + 1
    if sil[0]:
        starts = np.r_[0, starts]
    if sil[-1]:
        ends = np.r_[ends, len(sil)]
    m = min(len(starts), len(ends))
    lens = (ends[:m] - starts[:m]) * 0.01 if m else np.array([])
    odd = lens[(lens > 0.03) & (lens < 0.09)]      # 30~90ms 的"卡"感静音
    return len(hard), float(np.abs(d).max()), len(odd), (float(odd.mean()) if len(odd) else 0.0)


def main():
    for tag, af in (("A 现役", ""), ("B 加EQ", EQ)):
        y, sr = cut(af, tag.split()[0])
        st = stutter(y, sr)
        nhard, mx, nodd, avgodd = cuts(y, sr)
        print("%-8s 时长%.1fs  卡字重复=%d处 %s" % (
            tag, len(y) / float(sr), len(st),
            " ".join("%.1fs(重复%.0fms,相似%.3f)" % (t, lg * 1000, s)
                     for t, lg, s in st[:4]) if st else ""))
        print("         硬切(>60dB)=%d  最陡落差=%.1fdB  异常短静音(30~90ms)=%d 均长%.3fs"
              % (nhard, mx, nodd, avgodd))
    print("\nSTUCK_DONE")


if __name__ == "__main__":
    main()
