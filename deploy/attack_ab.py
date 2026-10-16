#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""「断句有异声」的量化：句间停顿的**起音/收尾斜率**

## 为什么查这个
两批员工一致：「都挺好，就断句有异声」「突然蹦出来『啊』的短音」。
已排除的假设（blip_ab.py 实测）：
  - 停顿中的孤立短发声段：我们 29.6 个/分 vs 真人原声 28.9 个/分 → **数量一样**，不是多了东西。
  - 语速：我们 7.55 字/秒 vs EL 锚点 7.57 → 一致（🔴 案例 merged.wav 是真人原声 5.6，别拿来当 EL 锚点）。

数量一样却听着"异"，那差异一定在**形状**：
真人在句子开头有呼吸/渐入，能量是**爬上来**的；
我们把元音直接**蹦**出来，能量是**跳上来**的 —— 这就是"突然蹦出啊"。

## 指标
对每个 ≥250ms 的停顿：
  - attack_ms ：停顿结束 → 能量从 floor+3dB 爬到 voiced−3dB 的耗时（越大越柔和）
  - release_ms：发声段末尾 → 能量从 voiced−3dB 落到 floor+3dB 的耗时
  - pre_breath：停顿**开始**后 150ms 内是否存在 −45~−30dB 的宽带噪声（真人的呼气/吸气）
  - onset_lvl：起音第一个峰值超出全篇发声均值多少 dB（>0 = 比正常更冲）

## 用法
/mnt/tts/projects/qwen3-tts/venv/bin/python attack_ab.py
"""
import os
import subprocess

import numpy as np
import soundfile as sf

TMP = "/mnt/tts/diag/_attack"
os.makedirs(TMP, exist_ok=True)
HOP = 0.005
PAUSE = 0.25           # 只统计 ≥250ms 的"句间"停顿

FILES = [
    ("ours_raw", "/mnt/tts/diag/xf5272/fem_raw.wav", "我们·裸推理"),
    ("ours_mp3", "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3", "我们·现役成品"),
    ("human",    "/mnt/tts/data/cases/52727ec43154/merged.wav", "真人原声（同文案）"),
]


def load16(path, tag):
    out = os.path.join(TMP, "%s.wav" % tag)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ac", "1", "-ar", "16000", out], check=True)
    return sf.read(out, dtype="float32")[0], 16000


def env(y, sr):
    hop = int(HOP * sr)
    n = len(y) // hop
    e = np.sqrt(np.array([(y[i * hop:(i + 1) * hop] ** 2).mean() for i in range(n)]))
    return 20 * np.log10(e + 1e-12)


def runs(mask):
    d = np.diff(mask.astype(np.int8))
    st = list(np.where(d == 1)[0] + 1)
    en = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        st.insert(0, 0)
    if mask[-1]:
        en.append(len(mask))
    m = min(len(st), len(en))
    return [(st[i], en[i]) for i in range(m)]


def cross(db, i0, i1, lvl, up=True):
    """在 [i0,i1) 内找首次穿到 lvl 的位置；找不到返回 None"""
    for i in range(i0, min(i1, len(db))):
        if (db[i] >= lvl) if up else (db[i] <= lvl):
            return i
    return None


def analyse(tag, path, desc):
    y, sr = load16(path, tag)
    db = env(y, sr)
    floor = np.percentile(db, 10)
    voiced = db[db > floor + 12].mean()
    lo, hi = floor + 3, voiced - 3

    on = db > floor + 6
    seg = runs(on)                      # 发声段 (start_idx, end_idx)
    gaps = []
    for k in range(len(seg) - 1):
        g = (seg[k][1], seg[k + 1][0])
        if (g[1] - g[0]) * HOP >= PAUSE:
            gaps.append((seg[k], g, seg[k + 1]))

    atk, rel, br, onl = [], [], [], []
    for (ps, pe), (gs, ge), (ns, ne) in gaps:
        # --- attack：停顿结束 ge → 爬到 hi ---
        i = cross(db, ge, min(ge + 120, len(db)), lo, up=True)
        j = cross(db, i if i is not None else ge, min(ge + 120, len(db)), hi, up=True) if i is not None else None
        if i is not None and j is not None:
            atk.append((j - i) * HOP * 1000)
            # 起音峰值：从 ge 起 60ms 内的最大电平
            onl.append(float(db[ge:min(ge + 12, len(db))].max()) - voiced)
        # --- release：前一发声段末尾 → 落到 lo ---
        i2 = cross(db, max(ps, pe - 120), pe, hi, up=False)
        j2 = cross(db, i2 if i2 is not None else pe, min(pe + 130, len(db)), lo, up=False) if i2 is not None else None
        if i2 is not None and j2 is not None:
            rel.append((j2 - i2) * HOP * 1000)
        # --- 停顿起始处是否有呼吸（−45~−30dB 的"中间电平"）---
        w = db[gs:min(gs + 30, ge)]
        if len(w) and (np.mean((w > -45) & (w < -30)) > 0.25):
            br.append(1)

    def med(x):
        return float(np.median(x)) if x else float("nan")

    print("\n=== %-9s %s" % (tag, desc))
    print("    时长 %.1fs  停顿(≥%.0fms) %d 处  发声均值 %.1fdB" % (len(y) / sr, PAUSE * 1000, len(gaps), voiced))
    print("    起音 attack 中位 %6.1f ms  (p25 %5.1f / p75 %5.1f)  n=%d"
          % (med(atk), np.percentile(atk, 25) if atk else 0,
             np.percentile(atk, 75) if atk else 0, len(atk)))
    print("    收尾 release 中位 %6.1f ms  (p25 %5.1f / p75 %5.1f)  n=%d"
          % (med(rel), np.percentile(rel, 25) if rel else 0,
             np.percentile(rel, 75) if rel else 0, len(rel)))
    print("    起音冲高 %+.1f dB（相对全篇发声均值，>0=比正常更冲）" % med(onl))
    print("    有呼吸铺垫的停顿 %d/%d = %.0f%%" % (len(br), len(gaps), 100.0 * len(br) / max(len(gaps), 1)))
    return dict(tag=tag, atk=med(atk), rel=med(rel), onl=med(onl),
                br=100.0 * len(br) / max(len(gaps), 1), n=len(gaps))


def main():
    res = []
    for tag, path, desc in FILES:
        if not os.path.exists(path):
            print("!! 缺 %s" % path)
            continue
        res.append(analyse(tag, path, desc))
    print("\n=== 汇总（attack 越大越柔和；真人 = 目标）===")
    print("%-10s %-10s %-11s %-11s %-10s" % ("文件", "停顿数", "attack ms", "release ms", "呼吸%"))
    for r in res:
        print("%-10s %-10d %-11.1f %-11.1f %-10.0f" % (r["tag"], r["n"], r["atk"], r["rel"], r["br"]))
    print("\nATTACKAB_DONE")


if __name__ == "__main__":
    main()
