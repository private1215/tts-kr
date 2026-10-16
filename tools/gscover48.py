# -*- coding: utf-8 -*-
"""gscover48.py —— GS 训练数据覆盖率体检：36 条案例到底用上了几条？

## 为什么要查
`tools/gs_prep_group.py --max-minutes 160` 是**按案例 id 排序**逐条累加音频、
到 160 分钟就 `break`。案例库全库 36 条合计 293.5 分钟：
- yohan（男）组 ~13 条 ≈ 108 min → 预算够，全用
- SOLA（女）组 ~22 条 ≈ 183 min → **160 min 预算会截断，末尾的案例直接丢掉**
被截断的部分是白扔的素材（GS 微调的唯一养料就是这 36 条 EL 案例）。

## 口径
完全复刻 gs_prep_group.main() 的顺序：
1. `embed_groups()` 聚类拿组（与训练脚本同一个函数，分组结果一致）
2. 去掉 hold 案例
3. CER 过滤（--exclude-cer）
4. 按 `sorted(cids)` 累加到 budget，超出即 break；最后一条按比例截断

用法： venv/bin/python gscover48.py [--max-minutes 160]
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/tools")

import mkcase_voices as mk  # noqa: E402

CASES_JSON = "/mnt/tts/data/cases.json"
CASES_DIR = "/mnt/tts/data/cases"
HOLD = {"SOLA": "00b6a45ad6aa", "yohan": "1a6c20a60481"}


def dur_of(cid):
    p = os.path.join(CASES_DIR, cid, "merged.wav")
    if not os.path.exists(p):
        return 0.0
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", p],
        capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=160.0)
    ap.add_argument("--exclude-cer", type=float, default=0.35)
    a = ap.parse_args()

    raw = json.load(open(CASES_JSON, encoding="utf-8"))
    lst = raw if isinstance(raw, list) else list(raw.get("cases", raw.values()))
    cases = {c["id"]: c for c in lst}
    g, meta = mk.embed_groups(cases, 0.75)

    print("全库 %d 条案例，合计 %.1f 分钟" % (len(cases), 293.5))
    print()
    for (nm, gd), cids in sorted(g.items(), key=lambda kv: -len(kv[1])):
        if len(cids) < 3:
            print("组 %s/%s：%d 条（<3，训练脚本会跳过）" % (nm, gd, len(cids)))
            continue
        hold = HOLD.get(nm) or sorted(cids)[0]
        train = [c for c in sorted(cids) if c != hold]
        # CER 过滤
        kept = []
        for cid in train:
            cs = [s.get("cer") or 0.0
                  for s in (cases[cid].get("segments") or [])]
            if cs and max(cs) > a.exclude_cer:
                print("  🚫 CER 剔除 %s（%.3f）" % (cid[:8], max(cs)))
                continue
            kept.append(cid)
        budget = a.max_minutes * 60.0
        got, used, dropped = 0.0, [], []
        for cid in kept:
            d = dur_of(cid)
            if got >= budget:
                dropped.append((cid, d, 0.0))
                continue
            if got + d > budget:
                k = budget - got
                used.append((cid, d, k))
                got = budget
                continue
            used.append((cid, d, d))
            got += d
        print("组 %s/%s：%d 条（hold=%s，CER 后 %d 条）"
              % (nm, gd, len(cids), hold[:8], len(kept)))
        print("  ✅ 进训练集 %d 条，%.1f / %.1f min"
              % (len([u for u in used]), got / 60.0, a.max_minutes))
        for cid, d, k in used:
            flag = "" if k >= d - 1 else "  ⚠️ 只用了 %.1f/%.1f min" % (k / 60, d / 60)
            print("     + %s %.1f min%s" % (cid[:8], d / 60.0, flag))
        if dropped:
            print("  ❌ 被预算丢掉 %d 条，合计 %.1f min："
                  % (len(dropped), sum(d for _, d, _ in dropped) / 60.0))
            for cid, d, _ in dropped:
                print("     - %s %.1f min" % (cid[:8], d / 60.0))
        print()


if __name__ == "__main__":
    main()
