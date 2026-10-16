#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定位员工听到的「啊 / 额」短促插入音

员工给的秒数（片段内相对时间）→ 成品绝对时间：
- 第 1 题（女声中段 90~120s）: 9s → **99s**、22s → **112s**
- 第 2 题（同一段）        : 5s → **95s**
- 第 3 题（女声尾部 165~185s）: 5s → **170s**

## 检测思路
"啊/额" = **孤立的短促发声段**：前后是静音，自身只有 80~400ms，
且能量/音高像个元音（不是辅音爆音）。
脚本先全篇扫出所有这类短段，再重点看员工点名的那几个时间点，
并**对照文案**看那几个位置是不是本来就有 아/어 之类的字。

用法：
  /mnt/tts/projects/qwen3-tts/venv/bin/python blip_find.py
"""
import os
import subprocess

import numpy as np
import soundfile as sf

SRC = "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"   # 女声现役成品
TXT = "/mnt/tts/data/cases/52727ec43154/full.txt"
N_CHARS = 1500          # 女声只生成前 1500 字
MARK = [95.0, 99.0, 112.0, 170.0]     # 员工点名的时间（成品绝对秒）

TMP = "/mnt/tts/diag/_blip"
os.makedirs(TMP, exist_ok=True)


def load16(path):
    out = os.path.join(TMP, "a.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ac", "1", "-ar", "16000", out], check=True)
    y, sr = sf.read(out, dtype="float32")
    return y, sr


def segs(y, sr):
    """返回 (静音段, 发声段) 列表，单位秒。"""
    hop = int(0.005 * sr)
    n = len(y) // hop
    e = np.sqrt(np.array([(y[i * hop:(i + 1) * hop] ** 2).mean() for i in range(n)]))
    db = 20 * np.log10(e + 1e-12)
    thr = np.percentile(db, 25) + 6          # 比底噪高 6dB 算有声
    v = db > thr
    d = np.diff(v.astype(int))
    st = np.where(d == 1)[0] + 1
    en = np.where(d == -1)[0] + 1
    if v[0]:
        st = np.r_[0, st]
    if v[-1]:
        en = np.r_[en, len(v)]
    m = min(len(st), len(en))
    return [(st[i] * 0.005, (en[i] - st[i]) * 0.005) for i in range(m)]


def txt_at(t, dur_total):
    """按时间比例估算文案位置（粗估，够定位用）。"""
    txt = open(TXT, encoding="utf-8").read().strip()[:N_CHARS]
    pos = int(t / dur_total * len(txt))
    return pos, txt[max(0, pos - 25):pos + 25].replace("\n", " ")


def main():
    y, sr = load16(SRC)
    dur = len(y) / float(sr)
    S = segs(y, sr)
    print("成品 %.1fs，发声段 %d 个\n" % (dur, len(S)))

    print("=== 全篇「孤立短发声段」（80~450ms，前后静音 ≥120ms）===")
    hits = []
    for i, (t, ln) in enumerate(S):
        if not (0.08 <= ln <= 0.45):
            continue
        prev_gap = t - (S[i - 1][0] + S[i - 1][1]) if i > 0 else 9.9
        next_gap = S[i + 1][0] - (t + ln) if i + 1 < len(S) else 9.9
        if prev_gap >= 0.12 and next_gap >= 0.12:
            hits.append((t, ln, prev_gap, next_gap))
    for t, ln, pg, ng in hits:
        flag = "  <<< 员工点名" if any(abs(t - m) < 1.5 for m in MARK) else ""
        pos, ctx = txt_at(t, dur)
        print("  %6.2fs 长%.3fs 前静音%.2fs 后静音%.2fs | 文案@%d: …%s…%s"
              % (t, ln, pg, ng, pos, ctx, flag))
    print("\n  共 %d 处（占全篇 %.2f 个/分钟）"
          % (len(hits), len(hits) / (dur / 60.0)))

    print("\n=== 员工点名时间点附近 ±1.5s 的发声段 ===")
    for m in MARK:
        near = [(t, ln) for t, ln in S if abs(t - m) < 1.5]
        pos, ctx = txt_at(m, dur)
        print("  %.0fs：%s" % (m, " / ".join("%.2fs(%.3fs)" % x for x in near) or "无"))
        print("        文案@%d: …%s…" % (pos, ctx))
    print("\nBLIP_DONE")


if __name__ == "__main__":
    main()
