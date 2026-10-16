#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第44轮（z12）裁决：句间停顿「地板抬起」—— 员工点名的三处全部复验

## 本轮依据（apply_silfix2.py，全部实测）
| | EL女声原声 | 我们现役 | 修后(地板-52) |
|---|---|---|---|
| 停顿帧中位 | −51.8dB | −51.9dB | −51.3dB |
| p5 | −60.5dB | −73.5dB | **−59.1dB** |
| p10 | −58.7dB | −62.4dB | **−57.9dB** |
| min | −98.5dB | **−240dB** | −86.5dB |
| <−100dB 帧 | 0 | 32 | **0** |
| 深谷(<−60dB,≥40ms) | 8 段 | 20 段 | 0 段 |
| 点名处最低 | — | 95.5s=−73 / 131.9s=−240 / 170.5s=−72 | **−59 / −59 / −60** |
| L1 Σz² | — | 32.2 | **27.8** |
| L1 平坦度 | — | +3.85σ | **+3.22σ** |
L1 闸门两项都**变好**，发声占比不变（−2.00σ → −2.01σ）。

## 员工点名 3 处 ↔ 深静音谷，3/3 精确命中
z43 题1 切自 90s 起 → 员工说"5 秒里有额" = 95s ✅
z43 题2 切自 130s 起 → 员工说"1 秒里有啊" = 131s ✅
z43 题3 切自 160s 起 → 员工说"10 秒有啊" = 170s ✅

## 机理
EL 的停顿"衰减到 −52dB 就平躺在房间底噪上"；
我们的停顿"穿过 −52 继续往下沉"，沉到 −73，甚至沉成数字零。
沉下去后语音再起来，落差从 30dB 变成 50~220dB → 听感上第一个音节「突然蹦出来」。
修法：加法式地板（只会抬高、绝不压低），把 <−58dB 的停顿段用**自身房间底噪**填到 −52dB。

## 前九次失败都别再重踩
① 软噪声门压拖尾 ② 填房间底噪(旧) ③ 句末淡出 ④ 元音起音柔化 ⑤ 分段压顶起
⑥ 迭代压顶起 ⑦ 连续平滑增益 ⑧ 句末突起压制 ⑨ 填静音(旧版两个 bug: 底噪用峰值归一
   导致实际只有 −68dB；淡入淡出没把原信号混回来反而挖坑)

## 护栏
本次只改**停顿区**，理论上不碰音节。但仍设一题专问停顿处有没有沙沙声/杂音 ——
万一填进去的底噪可闻，方案一样不能上线。

用法：/mnt/tts/projects/qwen3-tts/venv/bin/python mk_z44.py
"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
TMPD = "/mnt/tts/diag/_z44"
TGT_LUFS = -26.1
SR_CLIP = 24000

CUR = "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"   # 现役
FIX = "/mnt/tts/diag/_silfix2/f_52.mp3"                        # 句间地板抬起（−52dB）

CLIPS = [
    ("z44_a_mid", CUR, 90.0, 20.0),
    ("z44_b_mid", FIX, 90.0, 20.0),
    ("z44_a_lo", CUR, 130.0, 20.0),
    ("z44_b_lo", FIX, 130.0, 20.0),
    ("z44_a_tl", CUR, 160.0, 20.0),
    ("z44_b_tl", FIX, 160.0, 20.0),
]
SUFFIX = "20261016z12"


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


def two(iid, title, desc, a, b, la, lb, note, label_c="③ 两段差不多，听不出区别",
        label_d="④ 两段都像机器，都不像真人"):
    return {
        "id": iid, "dim": "女声·断句", "title": title, "desc": desc,
        "ref": None, "note": note,
        "options": [
            {"key": "A", "file": a, "label": la},
            {"key": "B", "file": b, "label": lb},
            {"key": "C", "file": None, "label": label_c},
            {"key": "D", "file": None, "label": label_d}]}


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
    made["z44_ctrl1"] = emit(clip(CUR, 60.0, 6.0), "z44_ctrl1")
    made["z44_ctrl2"] = emit(clip(CUR, 60.0, 6.0), "z44_ctrl2")

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

    D = ("两段读的是同一段话。\n"
         "**重点听：句子中间的「断口」** —— 有没有断得很突然、"
         "或者像「断了一下再接上」的感觉。\n听一遍就选，凭第一感觉。")
    DN = ("两段读的是同一段话。**这次只听一件事**：句子之间的**停顿 / 空白处**，\n"
          "有没有哪一段能听到沙沙声、嘶声、或者像底噪一样的杂音？\n\n"
          "（不用管像不像真人，也不用管断句。）")

    items = [
        two("z12_mid_%s" % SUFFIX,
            "【第 1 段】这两段，哪一段的断句更自然？",
            D, "z44_a_mid.mp3", "z44_b_mid.mp3",
            "① 第 1 段的断句更自然", "② 第 2 段的断句更自然",
            "裁决①：现役(①) vs 句间地板抬起(②)。\n"
            "🔴 **这一段第 5 秒就是员工第三次点名的「额」**（成品 95.50s，\n"
            "深静音谷 −73dB、125ms，而 EL 同口径停顿只到 −52dB）。B 已把该谷\n"
            "抬到 −52dB（复测：−59dB）。\n"
            "选② → 断句方案成立；选①/③/④ → 再回退改引擎路线。"),
        two("z12_lo_%s" % SUFFIX,
            "【第 2 段】这两段，哪一段的断句更自然？",
            D, "z44_a_lo.mp3", "z44_b_lo.mp3",
            "① 第 1 段的断句更自然", "② 第 2 段的断句更自然",
            "裁决②：换到 130~150s 复验。\n"
            "🔴 **第 1 秒是员工第三次点名的「啊」**（成品 131.28~131.37s，\n"
            "**数字零 −240dB**，75ms —— 全篇最严重的一处，A 原样保留）。\n"
            "B 已填成房间底噪。同一改动换个位置是否还成立。"),
        two("z12_tl_%s" % SUFFIX,
            "【第 3 段】这两段，哪一段的断句更自然？",
            D, "z44_a_tl.mp3", "z44_b_tl.mp3",
            "① 第 1 段的断句更自然", "② 第 2 段的断句更自然",
            "裁决③：换到 160~180s 复验。\n"
            "🔴 **第 10 秒是员工点名的第三处「啊」**（成品 169.93~170.12s，\n"
            "190ms、−72dB）。三处点名全部覆盖，避免上一轮「只修好一处」的问题。"),
        {"id": "z12_noise_%s" % SUFFIX, "dim": "女声·底噪护栏",
         "title": "【第 4 段】这两段的停顿处，有没有杂音？",
         "desc": DN, "ref": None,
         "note": "🔴 底噪护栏（素材同第 3 段，单独问一个维度）：\n"
                 "本次修法把停顿区从 −73/−240dB 抬到 −52dB，物理上是**加进了底噪**。\n"
                 "EL 的停顿本来就在 −52dB，所以理论上听不出；但万一「沙沙声」可闻，\n"
                 "方案一样否决 —— 哪怕第 1~3 题全赢。\n"
                 "正解参考：EL 原声同口径填完后中位 −51.3dB、p5 −59.1dB，与人声\n"
                 "主体差 35dB，正常音量下不可闻。",
         "options": [
             {"key": "A", "file": "z44_a_tl.mp3", "label": "① 第 1 段停顿处有杂音"},
             {"key": "B", "file": "z44_b_tl.mp3", "label": "② 第 2 段停顿处有杂音"},
             {"key": "C", "file": None, "label": "③ 两段都没有杂音"},
             {"key": "D", "file": None, "label": "④ 两段都有杂音"},
             {"key": "E", "file": None, "label": "⑤ 说不准 / 没听出来"}]},
        {"id": "z12_ctrl_same_%s" % SUFFIX, "dim": "答题有效性校验",
         "title": "【最后一题】这两段，听起来有区别吗？",
         "desc": "**这道题必须选一个**（选错也没关系）。\n\n"
                 "这两段是**同一段音频放了两遍**，内容完全一样。听 5 秒就够。",
         "ref": None, "note": "隐藏控制题：正解=C 两段一样。",
         "options": [
             {"key": "A", "file": "z44_ctrl1.mp3", "label": "A 明显有区别"},
             {"key": "B", "file": "z44_ctrl2.mp3", "label": "B 明显有区别"},
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
        print("  题目结构 OK（断句×3 + 底噪护栏1 + 控制1），全盲听")

    out = os.path.join(FB_DIR, "items.json")
    arc = os.path.join(FB_DIR, "items_roundz11.json")
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
    print("MANIFEST=" + json.dumps({k: md5(v) for k, v in made.items()},
                                   ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
