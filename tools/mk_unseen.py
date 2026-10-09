#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从**案例库自带的文案**生成阶段二（未见文案迁移）测试集。

案例上传时本来就是「语音 + txt 文案」成对入库的，所以案例库里现成就有大量
真实财经文案，不需要另外造文本。关键是**哪些没参与训练**：

  1. 训练说话人的 holdout 案例（bench_case）→ 它整条不进训练集，天然 unseen
  2. 同性别但**不是**该说话人的其它案例 → 完全没参与过该说话人的训练

反过来，参与训练的那些案例文案只能用于阶段一（同文案对标），
拿它们做阶段二就是 in-sample，会虚高。

输出格式（给 tools/ab2.py --stage 2 --texts-json 用）：
    [{"id":"799e4d","text":"...","src":"yohan","gender":"male","why":"holdout"}]

用法（服务器）：
    venv/bin/python tools/mk_unseen.py --speaker yohan --out /mnt/tts/diag/unseen.json
    venv/bin/python tools/mk_unseen.py --speaker yohan --holdout 799e4d7060ce \
        --out /mnt/tts/diag/unseen.json --only-holdout
"""
import os
import re
import sys
import json
import glob
import argparse

CASES_JSON = os.environ.get("TTS_CASES_JSON", "/mnt/tts/data/cases.json")
CASES_DIR = os.environ.get("TTS_CASES_DIR", "/mnt/tts/data/cases")


def case_text(cid):
    """取案例全文：优先 full.txt，否则按 segN.txt 顺序拼。"""
    d = os.path.join(CASES_DIR, cid)
    full = os.path.join(d, "full.txt")
    if os.path.exists(full):
        return open(full, encoding="utf-8").read()
    segs = sorted(glob.glob(os.path.join(d, "seg*.txt")),
                  key=lambda p: int(re.findall(r"seg(\d+)", p)[0]))
    return "\n".join(open(p, encoding="utf-8").read().strip() for p in segs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", default="yohan", help="训练用的说话人（要排除它的训练案例）")
    ap.add_argument("--gender", default="male", help="同性别的其它案例也纳入")
    ap.add_argument("--holdout", default="", help="holdout 案例 id（bench_case）")
    ap.add_argument("--only-holdout", action="store_true", help="只用 holdout 案例")
    ap.add_argument("--max-chars", type=int, default=0, help="截断每条文案到 N 字（0=不截断）")
    ap.add_argument("--out", default="/mnt/tts/diag/unseen.json")
    a = ap.parse_args()

    cases = json.load(open(CASES_JSON, encoding="utf-8"))
    cases = cases if isinstance(cases, list) else list(
        cases.get("cases", cases.values()))
    out = []

    def add(c, why):
        cid = c["id"]
        txt = case_text(cid).strip()
        if not txt:
            print(f"  !! {cid[:6]} 无文案，跳过")
            return
        if a.max_chars and len(txt) > a.max_chars:
            txt = txt[:a.max_chars]
        out.append(dict(id=cid[:6], text=txt, src=c.get("name") or "",
                        gender=c.get("gender") or "", why=why,
                        chars=len(txt)))
        print(f"  + {cid[:6]} [{why}] {len(txt)}字 src={c.get('name')}")

    for c in cases:
        cid = c["id"]
        name = (c.get("name") or "").strip()
        if a.holdout and cid == a.holdout:
            add(c, "holdout")
            continue
        if a.only_holdout:
            continue
        # 同性别但不属于训练说话人的案例 → 也没参与过训练
        if (c.get("gender") == a.gender) and name != a.speaker:
            add(c, "other_speaker_same_gender")

    if not out:
        sys.exit("没有可用的未见文案；检查 --speaker / --gender / --holdout")
    json.dump(out, open(a.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n✅ {len(out)} 条未见文案 -> {a.out}")
    for o in out:
        print(f"   {o['id']} {o['chars']}字 [{o['why']}]")


if __name__ == "__main__":
    main()
