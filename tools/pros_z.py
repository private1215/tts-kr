# -*- coding: utf-8 -*-
"""用 EL 案例的六维「情感丰富度」分布，给我们成品算偏离 sigma。

为什么这是比员工反馈更可靠的尺子：
    36 条案例全是 EL 生成的，prosody_report 的 A~F 已经算好存在 cases.json。
    过去只拿 1~2 条比对，从没当分布用。这里按性别建分布，再算 z。
    可量化、可回归、不依赖员工状态。

口径：必须和 cases.json 里存的值同脚本（prosody_report.measure）同参数，
      否则数字不可比。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")
import prosody_report as PR  # noqa: E402

CASES = "/mnt/tts/data/cases.json"
KEYS = list("ABCDEF")
NAME = {"A": "句内动态", "B": "音高跨度", "C": "能量动态",
        "D": "语速CV", "E": "句尾多样", "F": "停顿熵"}

# (标签, 路径, 性别)
# 🔴 改成命令行可覆盖（2026-10-14）：每次换候选都要改代码太慢。
#    用法： pros_z.py "v10:/mnt/tts/diag/x8197/m_v10.wav:male" ...
#    不传参数就用下面的默认表。
TARGETS = [
    ("EL原文8197", "/mnt/tts/data/cases/8197d9ac8a35/merged.wav", "male"),
    ("v7分窗变调", "/mnt/tts/diag/x8197/m_v7.wav", "male"),
    ("v5全局变调", "/mnt/tts/diag/x8197/m_v5.wav", "male"),
]
if len(sys.argv) > 1:
    TARGETS = []
    for _a in sys.argv[1:]:
        _p = _a.split(":")
        if len(_p) == 3:
            TARGETS.append((_p[0], _p[1], _p[2]))
        else:
            sys.stderr.write("忽略无法解析的参数：%s\n" % _a)


def dist(g):
    cs = json.load(open(CASES, encoding="utf-8")).values()
    rows = [c for c in cs if (c.get("gender") or "") == g]
    out = {}
    for k in KEYS:
        v = np.array([c["prosody"][k] for c in rows
                      if (c.get("prosody") or {}).get(k) is not None],
                     dtype=float)
        if len(v) >= 2:
            out[k] = (float(v.mean()), float(v.std(ddof=1)), len(v))
    return out


for g in ("male", "female"):
    base = dist(g)
    print("=" * 74)
    print("%s  EL 六维基线 n=%d" % (g, base["A"][2]))
    print("=" * 74)
    for k in KEYS:
        m, s, n = base[k]
        print("  %s %-6s 均值 %7.3f  sigma %6.3f   1sigma区间 %6.3f ~ %6.3f"
              % (k, NAME[k], m, s, m - s, m + s))

    for label, path, tg in TARGETS:
        if tg != g or not os.path.exists(path):
            continue
        try:
            r = PR.measure(path, label)
        except Exception as e:
            print("  FAIL", label, repr(e)[:100])
            continue
        print("\n--- %s ---" % label)
        zs = []
        for k in KEYS:
            if r.get(k) is None:
                continue
            m, s, _ = base[k]
            z = (r[k] - m) / (s if s > 1e-9 else 1e-9)
            zs.append((abs(z), k, r[k], z))
        zs.sort(reverse=True)
        for az, k, v, z in zs:
            flag = "[X]" if az > 2 else ("[!]" if az > 1 else "[ok]")
            print("  %-4s %s %-6s 我们 %7.3f  EL %7.3f  z=%+6.2f"
                  % (flag, k, NAME[k], v, base[k][0], z))
    print()
