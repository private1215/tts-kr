#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pause_probe3.py — 深挖：脏停顿明细 + raw/v2 位置匹配 + 男声 modulation 周期性"""
import json
from collections import Counter

d = json.load(open("/mnt/tts/diag/cross-20261013/pause_probe2.json"))
by = {r["label"]: r for r in d}

def dirty(rows):
    return [r for r in rows if r["tags"] != "clean"]

print("=== 女声 F_v2 仍脏的停顿明细 ===")
fd = dirty(by["F_v2"]["rows"])
fr = dirty(by["F_raw"]["rows"])
matched = [r for r in fd if any(abs(r["t"] - x["t"]) < 0.4 for x in fr)]
print(f"F_v2 dirty={len(fd)}  其中raw同位置也脏={len(matched)}  仅v2新增脏={len(fd)-len(matched)}")
for r in fd:
    m = [x for x in fr if abs(r["t"] - x["t"]) < 0.4]
    mm = ("raw_dur=%.3f raw_above=%.1f" % (m[0]["dur"], m[0]["max_above"])) if m else "raw干净/无匹配"
    print(f"  t={r['t']:7.2f} dur={r['dur']:.3f} above={r['max_above']:5.1f} flat={r['spec_flat']:.4f} {r['tags']:18s} {mm}")

print()
print("=== 女声 raw 脏但 v2 已干净的（清洗成功的）===")
fr_all = by["F_raw"]["rows"]
clean_v2_t = [r["t"] for r in by["F_v2"]["rows"]]
n_fixed = sum(1 for r in fr if any(abs(r["t"] - t) < 0.4 for t in clean_v2_t))
print(f"raw脏={len(fr)} 其中v2同位置变干净={n_fixed}")

print()
print("=== 男声 M_v4 脏停顿统计 ===")
md = dirty(by["M_v4"]["rows"])
c = Counter(r["tags"] for r in md)
print("M_v4 dirty:", len(md), dict(c))
print("dur 分布:", Counter(round(r["dur"], 1) for r in md).most_common(8))
above = sorted((r["max_above"] for r in md), reverse=True)
print("max_above top10:", [round(x, 1) for x in above[:10]])
print("max_above 中位数:", round(sorted(above)[len(above) // 2], 1))

print()
print("=== 男声 raw vs v2 vs v4 三版总览 ===")
for lab in ("M_raw", "M_v2", "M_v4"):
    r = by[lab]
    cc = Counter(x["tags"] for x in r["rows"])
    print(f"{lab:6s} pauses={r['n_pauses']:3d} {dict(cc)}")

print()
print("=== 男声 M_v4 modulation 型停顿的时间间隔（查周期性=loop接缝）===")
mod = [r["t"] for r in md if "modulation" in r["tags"]]
if len(mod) > 2:
    gaps = [round(mod[i + 1] - mod[i], 2) for i in range(len(mod) - 1)]
    print("间隔:", gaps[:40])
