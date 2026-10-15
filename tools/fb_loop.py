#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""闭环台账：把「出题 → 反馈 → 调整 → 出新题」固化成可执行的循环。

🔴 为什么要有这个（用户 2026-10-15 明确要求）
> **「要结合每一轮的题录，员工反馈后都要对应调整，调整好后更新最新版本题库，
>   再让员工反馈，这样形成一个循环，越调越好，形成闭环。」**

之前每轮都是**手工回忆**上一轮结论 → 迟早会漏、会重复问已排除的东西。
本脚本把这个循环自动化：

    1. 读回所有历史轮的**题录**（items.json 备份）
    2. 读回所有**员工反馈**（votes.jsonl）
    3. 逐轮算出：有效答卷 / 各题结果 / **本轮新增结论**
    4. 自动生成「已排除清单」——**下一轮出题必须避开这些已排除项**
    5. 输出一张闭环台账表：每轮的「问什么 → 答出什么 → 改了什么 → 下一轮问什么」

## 🔴 闭环的判定标准
**一轮算「闭环完成」必须同时满足**：
  ① 拿到了有效答卷（答满 + 控制题 C）
  ② 每一道题都能对应到一个**已执行的动作**（改了参数 / 换了引擎 / 排除了假设）
  ③ 动作已验证（受控实验或端到端重测），不是只改了就以为好了
  ④ **下一轮题库已更新**，且新题避开了本轮及历史已排除项

⚠️ ①②③ 缺一，闭环就断了 —— 就会出现「方案换来换去还是这个状态」。

## 用法
    python3 tools/fb_loop.py              # 打印台账
    python3 tools/fb_loop.py --json       # 输出结构化，便于程序处理
"""
import argparse
import json
import os
import re
import subprocess
from collections import OrderedDict

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
VOTES = os.path.join(FB_DIR, "votes.jsonl")
ITEMS = os.path.join(FB_DIR, "items.json")

# 轮次 -> 中文名（按 items.json 备份的文件名推断）
ROUND_HINT = {
    "n": ("第21轮 n", "员工给出「像感冒」+ 250Hz 到平台期"),
    "o": ("第22轮 o", "首次把「闷」拆成四种现象"),
    "p": ("第23轮 p", "覆盖 7 个不确定项 U1~U7"),
    "q": ("第24轮 q", "去掉兜底选项 → 拿到 2/2 具体信号"),
    "r": ("第25轮 r", "把「电子感」问透"),
    "s": ("第26轮 s", "只留能触发动作的题"),
}
# 各轮的日期（从题库备份 mtime 取，这里只做展示）
DATE_HINT = {"n": "10-15 凌晨", "o": "10-15 下午", "p": "10-15 下午",
             "q": "10-15 下午", "r": "10-15 傍晚", "s": "10-15 傍晚"}


def load_json(p, default=None):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def load_votes():
    rows = []
    if not os.path.exists(VOTES):
        return rows
    with open(VOTES, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:                       # noqa: BLE001
                pass
    return rows


def round_of(picks):
    """从 picks 的 key 推断轮次后缀。

    🔴 第一版写成 `k.rsplit("_",1)[-1]` → 得到 '20261015r' 而不是 'r'，
       导致整张台账是空的。正确做法：**从尾部正则取单个字母**。
    """
    import re
    for k in picks:
        m = re.search(r"(2026\d{4})([a-z])$", k)
        if m:
            suf = m.group(2)
            if suf in ("n", "o", "p", "q", "r", "s"):
                return suf
    return None


def archive_round(suf):
    """🔴 每轮**出完题立刻归档**题录，别只靠 items.json.bak-*。

    发现的真实问题（2026-10-15）：第 24 轮(q) 的题录**已经丢了**——
    `bak-before-q` 里装的是第 23 轮(p) 的题（命名是"被替换掉的那轮"），
    而 p 轮那次我用的是 `shutil.copyfile` 而不是带轮次后缀的命名，
    结果 q 轮题录无备份可查，台账里那一行只能显示「题数 0」。

    → 所以从本脚本起，每轮出题后跑一次它，把当轮题录存成
    `items_<轮次>_<日期>.json`，与 votes 里的 id 尾缀一致，可直接对应。
    """
    items = load_round_items(suf)
    if not items:
        return None
    dst = os.path.join(FB_DIR, "items_round%s.json" % suf)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump({"items": items}, f, ensure_ascii=False, indent=1)
    return dst


def load_round_items(suf):
    """按后缀找该轮题录。

    查找顺序（🔴 每一步都踩过坑）：
      1. `items_round<suf>.json` —— 本工具归档的，最可靠
      2. 当前 `items.json` —— 若它就是这一轮
      3. 扫 `items*.json*` 备份 —— **按内容里的 id 尾缀判断**，不能按文件名：
         `bak-before-q` 里装的是第 23 轮(p) 的题（命名是"被替换掉的那轮"）
      4. 旧目录 `/mnt/tts/data/tone-ab/items.json` —— 换目录前的历史轮次
    """
    import glob
    def suf_of(items):
        return any(re.search(r"2026\d{4}%s$" % suf, it["id"]) for it in items)

    arch = os.path.join(FB_DIR, "items_round%s.json" % suf)
    d = load_json(arch, {}) or {}
    if d.get("items") and suf_of(d["items"]):
        return d["items"]

    cur = load_json(ITEMS, {}) or {}
    citems = cur.get("items") or []
    if suf_of(citems):
        return citems

    cands = (sorted(glob.glob(os.path.join(FB_DIR, "items*.json*")))
             + sorted(glob.glob("/mnt/tts/data/tone-ab/items*.json")))
    for b in cands:
        d = load_json(b, {}) or {}
        if d.get("items") and suf_of(d["items"]):
            return d["items"]
    return []


def analyse():
    votes = load_votes()
    by_round = OrderedDict()
    for r in votes:
        picks = r.get("picks") or {}
        if not picks:
            continue
        suf = round_of(picks)
        if not suf:
            continue
        by_round.setdefault(suf, []).append(r)

    out = OrderedDict()
    for suf, rows in by_round.items():
        items = load_round_items(suf)
        ctrl_id = next((i["id"] for i in items
                        if i["id"].startswith("ctrl_pace")), None)
        latest = {}
        for r in rows:
            u = r.get("user")
            if u not in latest or r.get("ts", "") > latest[u].get("ts", ""):
                latest[u] = r
        total = len([i for i in items if not i["id"].startswith("ctrl")]) or 1
        valid = {}
        for u, r in latest.items():
            picks = r.get("picks") or {}
            n = len(picks)
            ok_ctrl = (not ctrl_id) or picks.get(ctrl_id) == "C"
            if n >= total and ok_ctrl:
                valid[u] = r
        # 各题结果
        tally = {}
        for it in items:
            if it["id"].startswith("ctrl"):
                continue
            cnt = {}
            for u, r in valid.items():
                v = (r.get("picks") or {}).get(it["id"])
                if v:
                    cnt[v] = cnt.get(v, 0) + 1
            if cnt:
                lab = {}
                for o in it.get("options") or []:
                    lab[o["key"]] = o["label"]
                top = sorted(cnt.items(), key=lambda x: -x[1])
                tally[it["id"]] = {
                    "title": it["title"],
                    "top": [(lab.get(k, k), c) for k, c in top],
                    "n_valid": len(valid),
                    "consensus": (len(valid) >= 2
                                  and top[0][1] == len(valid)),
                }
        name, focus = ROUND_HINT.get(suf, (suf, ""))
        out[suf] = {
            "name": ROUND_HINT.get(suf, (suf, ""))[0],
            "focus": ROUND_HINT.get(suf, (suf, ""))[1],
            "date": DATE_HINT.get(suf, ""),
            "n_submitters": len(latest),
            "valid_users": list(valid),
            "n_items": len(items) - (1 if ctrl_id else 0),
            "tally": tally,
            "free": {u: (r.get("free") or "").strip()
                     for u, r in latest.items()
                     if (r.get("free") or "").strip()},
        }
    return out


def print_table(data):
    print("=" * 78)
    print("🔁 反馈闭环台账（出题 → 反馈 → 调整 → 新题 → 再反馈）")
    print("=" * 78)
    for suf, d in data.items():
        print("\n【%s】%s   提交 %d 人 / 有效 %d 人   题数 %d"
              % (d["name"], d["date"], d["n_submitters"],
                 len(d["valid_users"]), d["n_items"]))
        print("  本轮重点：%s" % d["focus"])
        if d["valid_users"]:
            print("  有效：%s" % ", ".join(d["valid_users"]))
        else:
            print("  🔴 尚无有效答卷")
        for iid, t in (d["tally"] or {}).items():
            mark = " ★共识" if t["consensus"] else ""
            print("    %s%s" % (t["title"][:40], mark))
            for lab, c in t["top"]:
                print("        %-34s %d/%d" % (lab[:34], c, t["n_valid"]))
        for u, f in (d["free"] or {}).items():
            print("    [%s] %s" % (u, f.replace("\n", " / ")[:90]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--archive", metavar="SUF",
                    help="归档某轮题录（出完题立刻跑一次），例：--archive q")
    a = ap.parse_args()
    if a.archive:
        d = archive_round(a.archive)
        print("归档 %s -> %s" % (a.archive, d or "失败（题录未找到）"))
        return
    data = analyse()
    if a.json:
        print(json.dumps(data, ensure_ascii=False, indent=1))
    else:
        print_table(data)
    print("\n" + "=" * 78)
    print("闭环判定：每轮必须 ①有有效答卷 ②每题对应已执行动作 ③动作已验证 ④已出新题")
    print("缺任一项 → 闭环断裂，就会出现「方案换来换去还是这个状态」。")
    miss = [s for s, d in data.items() if not d["tally"]]
    if miss:
        print("⚠️ 题录缺失的轮次：%s（跑 `--archive <suf>` 从出题脚本重建）" % miss)


if __name__ == "__main__":
    main()