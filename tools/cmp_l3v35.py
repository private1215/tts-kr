#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L3v35 vs 案例（EL）标杆：全量指标对比，输出「还剩哪些差距」。

口径统一（不统一就作废，见项目规范）：
- 同一条案例的文案，我们走**生产 API** 全链路出片，EL 取 `cases/<id>/merged.wav`
- click：先归一到峰值 0.70 再判 |Δ|>=0.30，16k
- cps：字数与时长取自**同一条产出**（字数 = 案例 seg*.txt 拼接，含空格）
- 🔴 EL 自身运行间方差极大：click 2.56×、杂音 5.6×
  → 比值落在 <2.6× / <5.6× 区间内的都算「在 EL 噪声里」，不算真差距

用法：venv/bin/python tools/cmp_l3v35.py
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure53 import measure  # noqa: E402

PAIRS = [
    ("男声", "/mnt/tts/output/L3v35-male/L3v35-male-1.mp3",
     "/mnt/tts/data/cases/1a6c20a60481", "1a6c20a60481"),
    ("女声", "/mnt/tts/output/L3v35-female/L3v35-female-1.mp3",
     "/mnt/tts/data/cases/00b6a45ad6aa", "00b6a45ad6aa"),
]

# EL 自身运行间方差（同文本同嗓音两次录音的实测倍数）—— 判「真差距」的门槛
EL_VAR = {"click": 2.56, "noise": 5.6}

# 生产回检（jobs.json 里的 per-chunk CER）与完整性（longcov53 覆盖率）
PRECOMPUTED = {
    "男声": {"cer": 0.0508, "cov": 0.9441},
    "女声": {"cer": 0.0323, "cov": 0.9442},
}


def case_chars(cdir):
    segs = sorted(glob.glob(os.path.join(cdir, "seg*.txt")),
                  key=lambda p: int(re.findall(r"seg(\d+)", p)[0]))
    return len(" ".join(open(p, encoding="utf-8").read().strip()
                        for p in segs))


def row(name, ours, el, fmt="%.4g"):
    o, e = ours.get(name), el.get(name)
    if o is None or e is None:
        return "", ""
    r = (o / e) if e else float("nan")
    return fmt % o, fmt % e, ("%.2f×" % r) if e else "—"


def main():
    out = {}
    for gender, ours, cdir, cid in PAIRS:
        n = case_chars(cdir)
        el = os.path.join(cdir, "merged.wav")
        a = measure(ours, n)
        b = measure(el, n)
        # measure() 只打印不返回 cps；口径铁律：字数与时长取自同一条产出
        a["cps"] = n / a["dur"]
        b["cps"] = n / b["dur"]
        out[gender] = {"ours": a, "el": b, "chars": n}
        pre = PRECOMPUTED[gender]
        print("=" * 70)
        print("%s  案例 %s  文案 %d 字" % (gender, cid, n))
        print("%-18s %14s %14s %10s  %s"
              % ("指标", "我们 v3.5", "案例(EL)", "比值", "判定"))
        print("-" * 70)

        def line(label, key, fmt="%.4g", lower_better=True, var=None,
                 scale=1.0):
            o, e = a[key] * scale, b[key] * scale
            if not e:
                return
            r = o / e
            if var and (1.0 / var) < r < var:
                verdict = "在 EL 噪声内（<%.1f×）" % var
            elif abs(r - 1.0) < 0.05:
                verdict = "✅ 对齐"
            elif (r < 1.0) == lower_better:
                verdict = "✅ 我们更好"
            else:
                verdict = "⚠️ 我们更差"
            print("%-18s %14s %14s %10s  %s"
                  % (label, fmt % o, fmt % e, "%.2f×" % r, verdict))

        line("click %", "click", "%.4f", True, EL_VAR["click"], 100.0)
        line("谱质心 Hz", "centroid", "%.0f", None)
        line("中频 1k-3k %", "mid", "%.2f", None, None, 100.0)
        line("静音杂音 %", "noise", "%.2f", True, EL_VAR["noise"], 100.0)
        line("cps 字/秒", "cps", "%.2f", None)
        line("停顿中位 s", "pause_med", "%.3f", None)
        print("%-18s %14s %14s %10s  %s"
              % ("停顿变异 CV",
                 "%.2f" % a["pause_cv"], "%.2f" % b["pause_cv"],
                 "%.2f×" % (a["pause_cv"] / b["pause_cv"])
                 if b["pause_cv"] else "—", "越接近 1 越好"))
        print("%-18s %14d %14d" % ("停顿数", a["n_pause"], b["n_pause"]))
        print("%-18s %14.4f %14s" % ("CER(生产回检)", pre["cer"], "—"))
        print("%-18s %14.4f %14s" % ("覆盖率(完整性)", pre["cov"], "—"))
        print("%-18s %14.1f %14.1f %10s" % ("时长 s", a["dur"], b["dur"],
                                            "%.1f%%" % (100 * (a["dur"] / b["dur"] - 1))))
        print()
    json.dump({k: {"ours": v["ours"], "el": v["el"], "chars": v["chars"]}
               for k, v in out.items()},
              open("/mnt/tts/diag/cmp_l3v35.json", "w"), indent=1)
    print("已存 /mnt/tts/diag/cmp_l3v35.json")


if __name__ == "__main__":
    main()
