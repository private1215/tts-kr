#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第41轮（z9）「断句专项」定位题

## 员工原话（两批人一致）
    「都挺好，就断句有异声」
    「9秒和22秒那边突然蹦出来『啊』的短音」/「5秒有『额』」/「5秒有额、啊卡音特别短」

## 我已经查到的（客观）
| 指标 | 我们 | 对照 |
|---|---|---|
| 停顿中的孤立短发声段 | 29.6 个/分 | 真人 28.9 —— **一样，不是多了东西** |
| 语速 | 7.55 字/秒 | EL 锚点 7.57 —— 一致（🔴 案例 merged.wav 是真人 5.6，别拿它当 EL） |
| 句末 release | 117ms | 真人 22.5ms（我们慢 5 倍） |
| **死寂段（句间掉到数字零）** | **3.4% / 11.9 处每分** | **EL 长篇 1.4% / 2.7 处每分 ← 4.4 倍** |
| 前落差 / 中位时长 | 29dB / 155ms | EL 26dB / 148ms（接近，不是主因） |

试过并**否决**的修复（都写清楚，别重踩）：
- 软噪声门压句末拖尾：release 反而 117→152ms，整体电平掉 2.6dB 会加重吃字。
- 填房间底噪填死寂：死寂 3.32→0.30 处/分（有效），但 Σz² 32.1→34.7、
  平坦度 +3.83σ→−4.30σ（翻向）→ **指标回退，不上线**。

## 为什么还要问员工
客观指标走到尽头了，而**听感才是验收标准**。关键是先弄清员工听到的
"啊/额"是**哪一种**，因为两种的修法完全不同：
- **突然多出一个短音** → 模型多念了音节，要找生成侧原因/做删除
- **声音断了一下/卡了一下** → 句间死寂那一刀，要修停顿连续性

🔴🔴 普通人原则（用户第四次纠正，别违反）：
不做「9 段里哪段电子音最重」这种跨段落排名。改成**一段一题、逐段独立判断**，
并混入一段**员工没点名**的做隐藏对照，看我们的定位到底准不准。

## 用法
/mnt/tts/projects/qwen3-tts/venv/bin/python mk_z41.py
"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
TMPD = "/mnt/tts/diag/_z41"
TGT_LUFS = -26.1
SR_CLIP = 24000

OURS = "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"   # 女声现役成品 198.7s

# (名字, 起点, 时长, 说明)  s1/s2/s3 = 员工点名处；s4 = 隐藏对照（无点名、无死寂）
CLIPS = [
    ("z41_s1", 93.0, 10.0, "员工点名 95.5s / 99s"),
    ("z41_s2", 108.0, 10.0, "员工点名 112s"),
    ("z41_s3", 165.0, 10.0, "员工点名 170s"),
    ("z41_s4", 55.0, 10.0, "隐藏对照：员工没点名的普通段"),
]
SUFFIX = "20261016z9"


def lufs(p):
    r = subprocess.run(["ffmpeg", "-i", p, "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    v = [l for l in r.stderr.splitlines() if "I:" in l and "LUFS" in l]
    return float(v[-1].split("I:")[1].split("LUFS")[0].strip()) if v else None


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()[:8]


def clip(s0, dur, af=""):
    t = os.path.join(TMPD, "_c%s.wav" % hashlib.md5(
        ("%s%s%s" % (s0, dur, af)).encode()).hexdigest()[:6])
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", OURS, "-ss", "%.1f" % s0,
           "-ar", str(SR_CLIP), "-ac", "1", "-t", "%.1f" % dur]
    if af:
        cmd += ["-af", af]
    cmd += [t]
    subprocess.run(cmd, check=True)
    got = os.path.getsize(t) / 2.0 / SR_CLIP
    if got < dur - 0.5:
        sys.exit("切片越界：第%.0fs 起取%.0fs，实际仅%.1fs" % (s0, dur, got))
    return t


def emit(wav, name):
    g = TGT_LUFS - lufs(wav)
    out = os.path.join(FB_DIR, "%s.mp3" % name)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav,
                    "-af", "volume=%gdB" % g, "-b:a", "192k",
                    "-metadata", "title=%s" % name, out], check=True)
    back = lufs(out)
    print("  %-10s md5=%s  LUFS %6.1f %s"
          % (name, md5(out), back if back else 0,
             "OK" if back and abs(back - TGT_LUFS) < 1.0 else "!!爆表"))
    return out


def one(iid, title, desc, ref, note):
    """单音频判断题：一段一题，四选一（自然 / 多出音 / 断一下 / 说不清）"""
    return {
        "id": iid, "dim": "女声·断句", "title": title, "desc": desc,
        "ref": {"file": ref}, "note": note,
        "options": [
            {"key": "A", "file": None, "label": "① 听着自然，没注意到异样"},
            {"key": "B", "file": None, "label": "② 有异样：突然多出一个短音"},
            {"key": "C", "file": None, "label": "③ 有异样：声音断了一下 / 卡了一下"},
            {"key": "D", "file": None, "label": "④ 有异样，但说不清是哪种"}]}


def main():
    if not os.path.exists(OURS):
        print("缺素材：%s" % OURS)
        return 1
    os.makedirs(TMPD, exist_ok=True)
    made = {}
    print("=== 切片 + 归一（女声现役成品，10s × 4 段）===")
    for nm, s0, dur, _ in CLIPS:
        made[nm] = emit(clip(s0, dur), nm)
    made["z41_ctrl1"] = emit(clip(130.0, 6.0), "z41_ctrl1")
    made["z41_ctrl2"] = emit(clip(130.0, 6.0), "z41_ctrl2")

    print("\n=== md5 自检 ===")
    inv = {}
    for nm, p in made.items():
        inv.setdefault(md5(p), []).append(nm)
    bad = False
    for h, ns in inv.items():
        if len(ns) > 1 and "ctrl" not in ns[0]:
            print("  %-9s 撞车 -> %s" % (h, ns))
            bad = True
    if not bad:
        print("  无撞车（ctrl1/ctrl2 同 md5 是控制题的预期）")

    D = ("只听这 10 秒，凭第一感觉选。听一遍就选，不用反复听。\n"
         "如果没听出问题，就选① —— 选① 也是有效答案。")
    items = []
    for i, (nm, s0, dur, why) in enumerate(CLIPS, 1):
        items.append(one(
            "z9_seg%d_%s" % (i, SUFFIX),
            "【第 %d 段】这 10 秒听着有没有异样？" % i,
            D,
            "%s.mp3" % nm,
            "断句专项定位：%s\n"
            "服务端记位：成品 %.0f~%.0fs。\n"
            "选② → 模型多念了音节（内容侧问题），走「找生成原因 / 删伪音」路线。\n"
            "选③ → 句间那一刀（死寂 3.4%% vs EL 1.4%%），走「修停顿连续性」路线，\n"
            "      但填底噪已试过：Σz² 32.1→34.7、平坦度 +3.83σ→−4.30σ 回退，需换做法。\n"
            "选①/④ → 定位不准，回到波形再找。" % (why, s0, s0 + dur)))

    items.append({
        "id": "z9_ctrl_same_%s" % SUFFIX, "dim": "答题有效性校验",
        "title": "【最后一题】这两段，听起来有区别吗？",
        "desc": "**这道题必须选一个**（选错也没关系）。\n\n"
                "这两段是**同一段音频放了两遍**，内容完全一样。听 5 秒就够。",
        "ref": None, "note": "隐藏控制题：正解=C 两段一样。",
        "options": [
            {"key": "A", "file": "z41_ctrl1.mp3", "label": "A 明显有区别"},
            {"key": "B", "file": "z41_ctrl2.mp3", "label": "B 明显有区别"},
            {"key": "C", "file": None, "label": "C 没有区别，是一样的"}]})

    print("\n=== 题目自检 ===")
    ok = True
    for it in items:
        n = sum(1 for o in it["options"] if o.get("file"))
        r = (it.get("ref") or {}).get("file")
        for o in it["options"]:
            if o.get("file") and not os.path.exists(os.path.join(FB_DIR, o["file"])):
                print("  %s 缺文件 %s" % (it["id"], o["file"]))
                ok = False
        if n == 0 and not r:
            print("  %s 既无选项音频也无 ref -> 废题" % it["id"])
            ok = False
        if n == 1:
            print("  %s 单音频题必须改用 ref" % it["id"])
            ok = False
        blob = it["title"] + it["desc"] + "".join(o["label"] for o in it["options"])
        for kw in ("我们的", "EL", "标杆", "参考", "标准"):
            if kw in blob:
                print("  %s 泄漏身份（盲听题）：%s" % (it["id"], kw))
                ok = False
    if ok:
        print("  题目结构 OK（断句定位 4 + 控制 1），全盲听")

    out = os.path.join(FB_DIR, "items.json")
    arc = os.path.join(FB_DIR, "items_roundz8.json")
    cur = ""
    try:
        cur = io.open(out, encoding="utf-8").read()
    except OSError:
        pass
    if SUFFIX not in cur:
        shutil.copyfile(out, arc)
        print("  已归档上一轮 -> %s" % arc)
    else:
        print("  (items.json 已是本轮，跳过归档)")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"items": items}, f, ensure_ascii=False, indent=1)
    print("\n发布 %d 题 / %d 段音频 -> %s" % (len(items), len(made), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
