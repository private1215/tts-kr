#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""决定性对照：**同一文案**下，EL 在员工点名的那些秒数上是什么样？

## 为什么这是最强对照
`/mnt/tts/data/cases/52727ec43154/merged.wav` **就是 EL 原声**
（`ab/measure4.py` 白纸黑字：`("EL 原声 509s", ".../6cdd25bc75f2/merged.wav", 0.9628)`，
0.9628 = campplus 同人上限）。
而我们的成品 `prod_EL_female_52727ec43154.mp3` 用的是**同一份文案**。
两者语速几乎一样（EL 7.45 字/秒含空格、我们 7.55），所以**同一绝对秒 = 同一句话**。
→ 把同一秒的包络并排打出来，就能直接回答：
     EL 在这里有没有那个"额/啊"短音？
   有 → 是文案/律本身的东西，不是我们的问题
   没有 → 确实是我们多出来的，而且能看清形状差在哪

## 🔴 顺手纠正两个错误结论（写进记忆了，这里也留一份）
1. 案例库 `merged.wav` 是 **EL 原声**，不是真人原声。
   （`longtext_score.py` 里的 `HUMAN = .../merged.wav` 注释是误标。）
   语速口径必须用**含空格**的原始字符数：3911/525 = 7.45 ✅ 与 EL 锚点吻合；
   用去空白的 2952 会算出 5.62，那是错的。
2. `/mnt/tts/diag/ab/c_el_up.wav` 是「EL **上搬 1.094**」的变调版，
   包络被平滑过，它的 8.7 处/分**不能当 EL 基线**。
   真正的 EL 基线是 merged.wav 的 **30.5 处/分** → 我们 25.1 其实**比 EL 还少**，
   z43 那套「压顶起」方向是错的。

用法：/mnt/tts/projects/qwen3-tts/venv/bin/python blip_el_cmp.py
"""
import os
import subprocess

import numpy as np
import soundfile as sf

OURS = "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"
EL = "/mnt/tts/data/cases/52727ec43154/merged.wav"
TMP = "/mnt/tts/diag/_elcmp"
os.makedirs(TMP, exist_ok=True)
HOP = 0.005

# 员工历轮点名的位置（成品绝对秒）
MARKS = [(95.5, "题1/第2题 5s"), (99.0, "第1题 9s"), (112.0, "第1题 22s"),
         (131.9, "z43题2 1s"), (170.5, "第3题 10s")]


def load16(p, tag):
    o = os.path.join(TMP, "%s.wav" % tag)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ac", "1", "-ar", "16000", o], check=True)
    return sf.read(o, dtype="float32")[0], 16000


def db_of(y, sr):
    hop = int(HOP * sr)
    n = len(y) // hop
    e = np.sqrt(np.array([(y[i * hop:(i + 1) * hop] ** 2).mean() for i in range(n)]))
    return 20 * np.log10(e + 1e-12)


def main():
    yo, sr = load16(OURS, "ours")
    ye, _ = load16(EL, "el")
    do = db_of(yo, sr)
    de = db_of(ye, sr)
    fo = np.percentile(do, 10)
    fe = np.percentile(de, 10)
    print("我们 时长 %.1fs 底噪 %.1fdB" % (len(yo) / sr, fo))
    print("EL   时长 %.1fs 底噪 %.1fdB" % (len(ye) / sr, fe))
    print("（EL 全文 525s，我们只生成前 1500 字；语速接近 → 同一秒可对齐）")

    for t, note in MARKS:
        print("\n" + "=" * 74)
        print("【%.1fs】%s" % (t, note))
        for tag, d, fl in (("我们", do, fo), ("EL", de, fe)):
            a = int((t - 1.0) / HOP)
            b = int((t + 1.6) / HOP)
            seg = d[a:b]
            rows = []
            for i in range(0, len(seg), 10):      # 10 帧 = 50ms 一行
                rows.append("%.0f" % seg[i:i + 10].mean())
            print("  %-4s 50ms: %s" % (tag, " ".join(rows)))
        print("  相对底噪：上面数字减 %.0f(我们)/%.0f(EL) 即为高出量" % (fo, fe))

    print("\nBLIPELCMP_DONE")


if __name__ == "__main__":
    main()
