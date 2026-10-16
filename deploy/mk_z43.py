#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第43轮（z11）裁决：压掉「局部顶起的短音」= 员工说的"突然出来短音/额"

## 本轮唯一有 EL 正面证据的发现（apply_bump.py）
「局部顶起」= 一个 100~350ms 的发声段，电平比前后 400ms 内其他段的中位高 >8dB。
   真人原声  30.5 处/分钟（真人说话本来就有的重音起伏，听着自然）
   **EL 长篇  8.7 处/分钟**   ← 目标
   **我们     25.1 处/分钟 = EL 的 2.9 倍**
员工说的"5 秒处特别短的额音" = 成品 **95.50s**，精确认领：
   95.50s 235ms −23.4dB（上下文 −37.0dB，**高出 13.6dB**）—— 正是 blip_find 命中的那处。
z42 三题同源（90~110s），5 秒就是它，A/B 两段都有 → 员工当然选「差不多」。

## L1 闸门（probe_g.py，女声）
| 版本 | Σz² | 平坦度 | 发声占比 |
|---|---|---|---|
| 现役 | 32.1 | **+3.83σ** ← 女声已知短板 | −2.00σ |
| **压到 上下文+3dB** | **22.6** | **+0.20σ** ← **首次落进 EL 分布** | −1.89σ |
| 压到 上下文+6dB | 47.8 | +3.92σ（过头） | −3.80σ |
🔴 这不只是"顺一点"：**平坦度 +3.83σ → +0.20σ**，把记忆里"女声偏噪"的
   核心短板一次性修到 EL 分布里。Σz² 也 32.1→22.6（EL 分布 17~240）。

## 前四轮失败（都别重踩）
① 软噪声门压拖尾 ② 填房间底噪 ③ 句末淡出 ④ **元音起音柔化（z10 三票全「差不多/都不行」）**

## 护栏
压 83 处（约 8.4% 时长）可能把真实音节压轻 → 必须同时问「字清不清楚」。
🔴 断句题赢了、字清楚题输了 → 不能上线。

用法：/mnt/tts/projects/qwen3-tts/venv/bin/python mk_z43.py
"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
TMPD = "/mnt/tts/diag/_z43"
TGT_LUFS = -26.1
SR_CLIP = 24000

CUR = "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"     # 现役
FIX = "/mnt/tts/diag/_bump/b_t3.mp3"                            # 压到 上下文+3dB

# (输出名, 源, 起点, 时长)
CLIPS = [
    ("z43_a_mid",   CUR, 90.0, 20.0),
    ("z43_b_mid",   FIX, 90.0, 20.0),
    ("z43_a_clear", CUR, 130.0, 20.0),
    ("z43_b_clear", FIX, 130.0, 20.0),
    ("z43_a_tail",  CUR, 160.0, 20.0),
    ("z43_b_tail",  FIX, 160.0, 20.0),
]
SUFFIX = "20261016z11"


def lufs(p):
    r = subprocess.run(["ffmpeg", "-i", p, "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    v = [l for l in r.stderr.splitlines() if "I:" in l and "LUFS" in l]
    return float(v[-1].split("I:")[1].split("LUFS")[0].strip()) if v else None


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()[:8]


def clip(src, s0, dur):
    t = os.path.join(TMPD, "_c%s.wav" % hashlib.md5(
        ("%s%s%s" % (src, s0, dur)).encode()).hexdigest()[:6])
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ss", "%.1f" % s0,
                    "-ar", str(SR_CLIP), "-ac", "1", "-t", "%.1f" % dur, t], check=True)
    got = os.path.getsize(t) / 2.0 / SR_CLIP
    if got < dur - 0.5:
        sys.exit("切片越界：%s 第%.0fs 起，实际%.1fs" % (src, s0, got))
    return t


def emit(wav, name):
    g = TGT_LUFS - lufs(wav)
    out = os.path.join(FB_DIR, "%s.mp3" % name)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav,
                    "-af", "volume=%gdB" % g, "-b:a", "192k",
                    "-metadata", "title=%s" % name, out], check=True)
    back = lufs(out)
    print("  %-13s md5=%s  LUFS %6.1f %s"
          % (name, md5(out), back if back else 0,
             "OK" if back and abs(back - TGT_LUFS) < 1.0 else "!!爆表"))
    return out


def two(iid, title, desc, a, b, la, lb, note):
    return {
        "id": iid, "dim": "女声·短音", "title": title, "desc": desc,
        "ref": None, "note": note,
        "options": [
            {"key": "A", "file": a, "label": la},
            {"key": "B", "file": b, "label": lb},
            {"key": "C", "file": None, "label": "③ 两段差不多，听不出区别"},
            {"key": "D", "file": None, "label": "④ 两段都像机器，都不像真人"}]}


def main():
    for p in (CUR, FIX):
        if not os.path.exists(p):
            print("缺素材：%s" % p)
            return 1
    os.makedirs(TMPD, exist_ok=True)
    made = {}
    print("=== 切片 + 归一（20s × 6）===")
    for nm, src, s0, dur in CLIPS:
        made[nm] = emit(clip(src, s0, dur), nm)
    made["z43_ctrl1"] = emit(clip(CUR, 60.0, 6.0), "z43_ctrl1")
    made["z43_ctrl2"] = emit(clip(CUR, 60.0, 6.0), "z43_ctrl2")

    print("\n=== md5 自检 ===")
    inv = {}
    for nm, p in made.items():
        inv.setdefault(md5(p), []).append(nm)
    bad = False
    for h, ns in inv.items():
        if len(ns) > 1 and "ctrl" not in ns[0]:
            print("  %-9s 撞车 -> %s" % (h, ns))
            bad = True
    print("  无撞车（ctrl1/ctrl2 除外）" if not bad else "  !! 有撞车")

    D = ("两段内容一样，第 2 段做了一点处理。\n"
         "**重点听：有没有突然冒出来的、特别短的一声**（像「额」）。\n"
         "听一遍就选，凭第一感觉。")
    DC = ("两段内容一样。**不用管像不像真人、也不用管短音**，只听字：\n"
          "有没有被吃掉、含不含糊。")

    items = [
        two("z11_mid_%s" % SUFFIX,
            "【第 1 段】这两段，哪一段更顺？",
            D, "z43_a_mid.mp3", "z43_b_mid.mp3",
            "① 第 1 段更顺", "② 第 2 段更顺",
            "裁决①：现役(①) vs 压顶起短音(②)。\n"
            "🔴 **这一段 5 秒处就是员工两次点名的那个「额」音**（成品 95.50s，\n"
            "235ms、高出上下文 13.6dB）。两段里 A 原样、B 已压平。\n"
            "选② → 短音消失，方案成立；选①/③ → 压顶起无效，转引擎路线。"),
        two("z11_clear_%s" % SUFFIX,
            "【第 2 段】这两段，哪一段的字听得更清楚？",
            DC, "z43_a_clear.mp3", "z43_b_clear.mp3",
            "① 第 1 段字更清楚", "② 第 2 段字更清楚",
            "🔴 防吃字护栏（换到 130~150s 新位置，避免疲劳）：\n"
            "压顶起会把短音压低，若那些是真实音节就会「吃字」。\n"
            "选① → 方案否决，哪怕第 1 题赢了也不能上线。"),
        two("z11_tail_%s" % SUFFIX,
            "【第 3 段】这两段，哪一段更顺？",
            D, "z43_a_tail.mp3", "z43_b_tail.mp3",
            "① 第 1 段更顺", "② 第 2 段更顺",
            "裁决②：换到 160~180s 复验（该区间也有密集顶起：\n"
            "161.0 / 161.6 / 164.9 / 167.1 / 170.5s）。同一改动换个位置是否还成立。"),
        {"id": "z11_ctrl_same_%s" % SUFFIX, "dim": "答题有效性校验",
         "title": "【最后一题】这两段，听起来有区别吗？",
         "desc": "**这道题必须选一个**（选错也没关系）。\n\n"
                 "这两段是**同一段音频放了两遍**，内容完全一样。听 5 秒就够。",
         "ref": None, "note": "隐藏控制题：正解=C 两段一样。",
         "options": [
             {"key": "A", "file": "z43_ctrl1.mp3", "label": "A 明显有区别"},
             {"key": "B", "file": "z43_ctrl2.mp3", "label": "B 明显有区别"},
             {"key": "C", "file": None, "label": "C 没有区别，是一样的"}]},
    ]

    print("\n=== 题目自检 ===")
    ok = True
    for it in items:
        n = sum(1 for o in it["options"] if o.get("file"))
        for o in it["options"]:
            if o.get("file") and not os.path.exists(os.path.join(FB_DIR, o["file"])):
                print("  %s 缺文件 %s" % (it["id"], o["file"]))
                ok = False
        if n == 0 and not (it.get("ref") or {}).get("file"):
            print("  %s 无音频 -> 废题" % it["id"])
            ok = False
        if n == 1:
            print("  %s 单音频题必须改用 ref" % it["id"])
            ok = False
        blob = it["title"] + it["desc"] + "".join(o["label"] for o in it["options"])
        for kw in ("我们的", "EL", "标杆", "参考", "标准"):
            if kw in blob:
                print("  %s 泄漏身份：%s" % (it["id"], kw))
                ok = False
    if ok:
        print("  题目结构 OK（短音中段1 + 字清楚1 + 短音尾部1 + 控制1），全盲听")

    out = os.path.join(FB_DIR, "items.json")
    arc = os.path.join(FB_DIR, "items_roundz10.json")
    try:
        cur = io.open(out, encoding="utf-8").read()
    except OSError:
        cur = ""
    if SUFFIX not in cur:
        shutil.copyfile(out, arc)
        print("  已归档上一轮 -> %s" % arc)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"items": items}, f, ensure_ascii=False, indent=1)
    print("\n发布 %d 题 / %d 段音频 -> %s" % (len(items), len(made), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
