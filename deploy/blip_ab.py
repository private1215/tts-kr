#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""「断句有异声」三方对照：我们裸推理 / 我们现役成品 / EL 原文（同文案）

两位员工一致：「都挺好，就断句有异声」。要判定：
  1. 这个"异声"是**模型原生**还是**后处理引入**？  → 裸推理 vs 成品
  2. 这个"异声"是**我们特有**还是 EL 也有？        → 成品 vs EL 原文

## 指标 A：孤立短发声段（停顿中的"啊/额"）
  静音 ≥120ms 包围、自身 60~500ms 的发声段 → 个/分钟

## 指标 B：停顿纯净度
  所有 ≥150ms 的停顿，统计内部"超出底噪 6/10dB"的时间占比。
  停顿越不干净，越容易听出"断句有异声"。
  口径统一：底噪 = 全篇 db 的 10 分位（近似静音地板）

用法：/mnt/tts/projects/qwen3-tts/venv/bin/python blip_ab.py
环境变量：ONLY=fem_raw 只跑一个
"""
import os
import subprocess

import numpy as np
import soundfile as sf

TMP = "/mnt/tts/diag/_blipab"
os.makedirs(TMP, exist_ok=True)

BASE = "/mnt/tts/diag/xf5272"
FILES = [
    ("ours_raw", "%s/fem_raw.wav" % BASE, "我们·裸推理（未后处理）"),
    ("ours_eq",  "%s/fem_witheq.wav" % BASE, "我们·后处理+EQ"),
    ("ours_mp3", "%s/prod_EL_female_52727ec43154.mp3" % BASE, "我们·现役成品(mp3)"),
    ("EL",       "/mnt/tts/data/cases/52727ec43154/merged.wav", "EL 原文（同文案）"),
]

HOP = 0.005
GAP = 0.12          # 孤立判定：前后静音 ≥120ms
MINLEN, MAXLEN = 0.06, 0.50


def load16(path, tag):
    out = os.path.join(TMP, "%s.wav" % tag)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ac", "1", "-ar", "16000", out], check=True)
    y, sr = sf.read(out, dtype="float32")
    return y, sr


def env_db(y, sr):
    hop = int(HOP * sr)
    n = len(y) // hop
    e = np.sqrt(np.array([(y[i * hop:(i + 1) * hop] ** 2).mean() for i in range(n)]))
    return 20 * np.log10(e + 1e-12), n


def runs(mask):
    """mask: bool 数组 → [(start_idx, len)]"""
    d = np.diff(mask.astype(np.int8))
    st = list(np.where(d == 1)[0] + 1)
    en = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        st.insert(0, 0)
    if mask[-1]:
        en.append(len(mask))
    m = min(len(st), len(en))
    return [(st[i], en[i] - st[i]) for i in range(m)]


def analyse(tag, path, desc):
    y, sr = load16(path, tag)
    dur = len(y) / sr
    db, n = env_db(y, sr)
    floor = np.percentile(db, 10)          # 静音地板
    voiced_ref = db[db > floor + 12].mean()  # 发声段平均电平

    # ---- A：孤立短发声段 ----
    on = db > floor + 6
    seg = runs(on)
    blips = []
    for i, (s, L) in enumerate(seg):
        ln = L * HOP
        if not (MINLEN <= ln <= MAXLEN):
            continue
        pre = (seg[i - 1][1] * HOP) if i > 0 else 999.0
        post = (seg[i + 1][1] * HOP) if i < len(seg) - 1 else 999.0
        if pre >= GAP and post >= GAP:
            lvl = 20 * np.log10(np.sqrt((y[int(s * HOP * sr):int((s + L) * HOP * sr)] ** 2).mean()) + 1e-12)
            blips.append((s * HOP, ln, lvl))

    # ---- B：停顿纯净度 ----
    off = ~on
    pauses = [(s * HOP, L * HOP) for s, L in runs(off) if L * HOP >= 0.15]
    if pauses:
        tot = sum(L for _, L in pauses)
        dirty6 = dirty10 = 0.0
        for s, L in pauses:
            a, b = int(s * sr), int((s + L) * sr)
            d = db[int(s / HOP):int((s + L) / HOP)]
            dirty6 += float((d > floor + 6).mean()) * L
            dirty10 += float((d > floor + 10).mean()) * L
        p6, p10 = 100 * dirty6 / tot, 100 * dirty10 / tot
    else:
        tot, p6, p10 = 0.0, 0.0, 0.0

    print("\n=== %-10s %s" % (tag, desc))
    print("    时长 %.1fs  静音地板 %.1fdB  发声均值 %.1fdB" % (dur, floor, voiced_ref))
    print("    [A] 孤立短发声段 %2d 处 = %.2f 个/分钟（均值时长 %.3fs，电平 %.1fdB）"
          % (len(blips), len(blips) / (dur / 60.0),
             np.mean([b[1] for b in blips]) if blips else 0,
             np.mean([b[2] for b in blips]) if blips else 0))
    print("    [B] ≥150ms 停顿 %d 段 共 %.1fs | 停顿内超地板6dB 占 %.1f%% | 超10dB 占 %.1f%%"
          % (len(pauses), tot, p6, p10))
    for t, ln, lvl in blips:
        print("        %7.2f  %.3fs  %.1fdB" % (t, ln, lvl))
    return dict(tag=tag, dur=dur, n=len(blips), rate=len(blips) / (dur / 60.0),
                p6=p6, p10=p10, floor=floor, blips=blips)


def main():
    only = os.environ.get("ONLY", "")
    res = []
    for tag, path, desc in FILES:
        if only and only != tag:
            continue
        if not os.path.exists(path):
            print("!! 缺 %s" % path)
            continue
        res.append(analyse(tag, path, desc))

    print("\n=== 汇总（个/分钟 = 停顿中孤立短音密度；越低越干净）===")
    print("%-10s %-8s %-10s %-10s %-10s %-10s" % ("文件", "时长", "密度个/分", "超6dB%", "超10dB%", "静音地板"))
    for r in res:
        print("%-10s %-8.1f %-10.2f %-10.1f %-10.1f %-10.1f"
              % (r["tag"], r["dur"], r["rate"], r["p6"], r["p10"], r["floor"]))
    print("\nBLIPAB_DONE")


if __name__ == "__main__":
    main()
