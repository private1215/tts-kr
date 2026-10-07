# -*- coding: utf-8 -*-
"""分块稳定性回归：同一篇文案在 cps 抖动范围内，块数必须不变。

为什么需要它
------------
2026-10-07 发现两件事：
1. `8197d9` 分 3 段不是 bug —— 它 4489 字 ≈ 530 秒，装不进 2×250 秒，
   而**案例自己是 2 段只是当时人工截成两半上传的**，不该拿来当规则。
2. 但 `ba24c0` 的块数会**随 cps 抖动在 2/3 之间跳**（r21c 3 块、r22 2 块），
   这是真 bug：贪心把 19 字的打包余数当成了一个块，引爆了「尾块过短→等分」。

本脚本扫一遍真实案例文案 × cps 抖动区间，块数一旦变化就是回归。

用法（服务器）：
    cd /mnt/tts/projects/qwen3-tts && venv/bin/python tools/test_chunk.py
"""
import os
import glob
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.pipeline import (split_by_seconds, chunk_cps, chunk_limit_chars,  # noqa: E402
                          CHUNK_MAX_SECONDS, CHUNK_CPS_LOCK_RATIO)

CASES_DIR = os.environ.get("TTS_CASES_DIR", "/mnt/tts/data/cases")
# 男 8.48 / 女 7.27 是当前值；扫 ±3% 覆盖 cps 滑动平均的正常抖动
CASES = [
    ("1a6c20a60481", "male", 8.48),
    ("00b6a45ad6aa", "female", 7.27),
    ("ba24c0925311", "male", 8.48),
    ("8197d9ac8a35", "male", 8.48),
    ("6cdd25bc75f2", "male", 8.48),
]


def case_text(cid):
    d = os.path.join(CASES_DIR, cid)
    full = os.path.join(d, "full.txt")
    if os.path.exists(full):
        return open(full, encoding="utf-8").read()
    import re
    segs = sorted(glob.glob(os.path.join(d, "seg*.txt")),
                  key=lambda p: int(re.findall(r"seg(\d+)", p)[0]))
    return "\n".join(open(p, encoding="utf-8").read().strip() for p in segs)


def main():
    print(f"预算 {CHUNK_MAX_SECONDS:.0f}s   "
          f"（案例自身块长：242~294s，2 段）\n")
    bad = 0
    for cid, gender, cps in CASES:
        t = case_text(cid)
        if not t.strip():
            print(f"{cid}: 取不到文案，跳过")
            continue
        # 模拟线上：voices.json 里锁着一个分块语速，实测 cps 在 ±3% 内漂
        meta = {"cps_lock": cps}
        rows = []
        for f in (0.97, 0.985, 1.0, 1.015, 1.03):
            c = cps * f
            locked, _ = chunk_cps(meta, c)
            assert abs(locked - cps) < 1e-9 or abs(c - cps) / cps > CHUNK_CPS_LOCK_RATIO
            ch = split_by_seconds(t, cps=locked, speed=1.0)
            secs = [round(len(x) / locked) for x in ch]
            rows.append((round(c, 3), len(ch), secs))
        counts = set(r[1] for r in rows)
        flag = "OK " if len(counts) == 1 else "🔴 跳动"
        if len(counts) != 1:
            bad += 1
        print(f"{cid[:6]} {gender:<6} {len(t):>5}字  {flag}  块数={sorted(counts)}")
        for c, n, secs in rows:
            print(f"        cps={c:<6} -> {n} 块 {secs}")
    print()
    if bad:
        print(f"🔴 {bad} 个案例块数不稳定")
        return 1
    print("✅ 全部案例在 cps ±3% 抖动下块数稳定")
    return 0


if __name__ == "__main__":
    sys.exit(main())
