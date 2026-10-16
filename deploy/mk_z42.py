#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第42轮（z10）裁决：**元音起音柔化**（针对「全部突然出来短音」）

## 员工 z9 实测（两票有效，4 段全中，含隐藏对照段）
    디아      ：4/4 「声音断了一下 / 卡了一下」
    24124112  ：4/4 「突然多出一个短音」，自由文本「全部突然 出来 短音」
→ **全篇普遍**，不是 95/99/112/170 那几个点。

## 本轮候选（apply_onset.py）
只对**元音起音**做淡入（辅音起音不动，否则加重吃字）：
  停顿 ≥80ms 后的起音，前 15ms 的 ≥4kHz 能量占比 ≤0.15 判为元音 → 淡入。
  实测柔化元音起音 74 处 / 保留辅音起音 67 处，时长严格不变。

L1 闸门（probe_g.py，女声）：
| 版本 | Σz² | 平坦度 | 发声占比 |
|---|---|---|---|
| 现役 | 32.1 | +3.83σ | −2.00σ |
| 温和档 fade40 gain0.45 | **32.5** | +4.02σ | −1.90σ |
| 明显档 fade60 gain0.28 | **32.4** | +4.16σ | **−1.63σ** |
→ 变化 <1、平坦度**没翻向**（对比填底噪方案：Σz² +2.6~4.3 且 +3.83σ→−4.30σ 翻向）
→ 这是**四次尝试里第一个过闸的**。

## 前三条失败路线（别重踩，都记在脚本头里）
软噪声门压拖尾 / 填房间底噪 / 句末淡出。

## 用法
/mnt/tts/projects/qwen3-tts/venv/bin/python mk_z42.py
"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
TMPD = "/mnt/tts/diag/_z42"
TGT_LUFS = -26.1
SR_CLIP = 24000

CUR = "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"          # 现役
SOFT = "/mnt/tts/diag/_onset/o_40_0.15_0.45.mp3"                      # 温和档
HARD = "/mnt/tts/diag/_onset/o_60_0.15_0.28.mp3"                      # 明显档

# (输出名, 源文件, 起点, 时长)
CLIPS = [
    ("z42_a_cur",  CUR,  90.0, 20.0),
    ("z42_b_soft", SOFT, 90.0, 20.0),
    ("z42_c_hard", HARD, 90.0, 20.0),
]
SUFFIX = "20261016z10"


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
        sys.exit("切片越界：%s 第%.0fs 起取%.0fs，实际%.1fs" % (src, s0, dur, got))
    return t


def emit(wav, name):
    g = TGT_LUFS - lufs(wav)
    out = os.path.join(FB_DIR, "%s.mp3" % name)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav,
                    "-af", "volume=%gdB" % g, "-b:a", "192k",
                    "-metadata", "title=%s" % name, out], check=True)
    back = lufs(out)
    print("  %-12s md5=%s  LUFS %6.1f %s"
          % (name, md5(out), back if back else 0,
             "OK" if back and abs(back - TGT_LUFS) < 1.0 else "!!爆表"))
    return out


def two(iid, title, desc, a, b, la, lb, note):
    return {
        "id": iid, "dim": "女声·断句", "title": title, "desc": desc,
        "ref": None, "note": note,
        "options": [
            {"key": "A", "file": a, "label": la},
            {"key": "B", "file": b, "label": lb},
            {"key": "C", "file": None, "label": "③ 两段差不多，听不出区别"},
            {"key": "D", "file": None, "label": "④ 两段都像机器，都不像真人"}]}


def main():
    for p in (CUR, SOFT, HARD):
        if not os.path.exists(p):
            print("缺素材：%s" % p)
            return 1
    os.makedirs(TMPD, exist_ok=True)
    made = {}
    print("=== 切片 + 归一（20s × 3）===")
    for nm, src, s0, dur in CLIPS:
        made[nm] = emit(clip(src, s0, dur), nm)
    made["z42_ctrl1"] = emit(clip(CUR, 130.0, 6.0), "z42_ctrl1")
    made["z42_ctrl2"] = emit(clip(CUR, 130.0, 6.0), "z42_ctrl2")

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
        print("  无撞车（ctrl1/ctrl2 除外）")

    D1 = ("两段内容一样，只是第 2 段做了调整。\n"
          "重点听：**句子和句子交界的地方**，有没有突然冒出来的短音。\n"
          "听一遍就选，凭第一感觉。")
    D2 = ("两段内容一样。**不用管像不像真人**，只听字：\n"
          "有没有被吃掉、含不含糊。")

    items = [
        two("z10_soft_%s" % SUFFIX,
            "【断句·第 1 组】这两段，哪一段交界处更顺？",
            D1, "z42_a_cur.mp3", "z42_b_soft.mp3",
            "① 第 1 段更顺", "② 第 2 段更顺",
            "裁决：现役(①) vs 元音起音柔化·温和档(②，fade40/gain0.45)。\n"
            "L1：Σz² 32.1→32.5、平坦度 +3.83σ→+4.02σ（未翻向）、发声占比 −2.00→−1.90σ。\n"
            "选② → 起音柔化有效，上线温和档；\n"
            "选①/③ → 柔化无效或过头，断句问题不在起音，转引擎/微调路线。"),
        two("z10_hard_%s" % SUFFIX,
            "【断句·第 2 组】这两段，哪一段交界处更顺？",
            D1, "z42_a_cur.mp3", "z42_c_hard.mp3",
            "① 第 1 段更顺", "② 第 2 段更顺",
            "裁决：现役(①) vs 元音起音柔化·明显档(②，fade60/gain0.28)。\n"
            "L1：Σz² 32.1→32.4、平坦度 +4.16σ、发声占比 −2.00σ→**−1.63σ（最优）**。\n"
            "若本组选②而第 1 组选③ → 说明柔化要够狠才听得出，采用明显档。"),
        two("z10_clear_%s" % SUFFIX,
            "【字清不清楚】这两段，哪一段的字听得更清楚？",
            D2, "z42_a_cur.mp3", "z42_b_soft.mp3",
            "① 第 1 段字更清楚", "② 第 2 段字更清楚",
            "🔴 防吃字护栏：起音淡入会削弱开头音节，可能加重「吃字」。\n"
            "选②/③ → 柔化不伤字，可上线；\n"
            "选① → 柔化把字吃掉了，本方案否决（哪怕断句题赢了也不能上）。"),
        {"id": "z10_ctrl_same_%s" % SUFFIX, "dim": "答题有效性校验",
         "title": "【最后一题】这两段，听起来有区别吗？",
         "desc": "**这道题必须选一个**（选错也没关系）。\n\n"
                 "这两段是**同一段音频放了两遍**，内容完全一样。听 5 秒就够。",
         "ref": None, "note": "隐藏控制题：正解=C 两段一样。",
         "options": [
             {"key": "A", "file": "z42_ctrl1.mp3", "label": "A 明显有区别"},
             {"key": "B", "file": "z42_ctrl2.mp3", "label": "B 明显有区别"},
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
                print("  %s 泄漏身份（盲听题）：%s" % (it["id"], kw))
                ok = False
    if ok:
        print("  题目结构 OK（断句温和1 + 断句明显1 + 字清楚1 + 控制1），全盲听")

    out = os.path.join(FB_DIR, "items.json")
    arc = os.path.join(FB_DIR, "items_roundz9.json")
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
