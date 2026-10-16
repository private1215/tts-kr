#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第三十五轮题库：普通人角度（用户第四次纠正后定稿）

## 为什么废掉上一轮（mk_z34.py 的 9 段排名题）
> 「正常人听 9 个语音，你让选一个电子音最重的，不同的音感不一样，
>   你这不是在逗人玩吗，普通人的角度出题」

9 段是**不同内容**的段落，音感本来就不同，让普通人跨内容排名
「哪段最电子」= 逼人做做不了的判断。
（唯一收获：员工选了「每段都一样」——电子感是全局的，EL 对照选「都没有」。）

## 普通人能做什么（历史证明）
- ✅ **绝对判断**：一段音频「像不像真人」——员工脱口而出「挺接近，还是有电子音」
- ✅ **二选一裁决**（同内容 A/B）——第 26 轮 3/3 全员一致
- ❌ 跨内容排名 / 反应速度 / 打字描述

## 这一轮的题（7 题，每题只点一下）
| 题 | 形式 | 用途 |
|---|---|---|
| 1. 发客户基线 | 30s A/B（A=我们的 B=EL C=都不行）| 长期锚点，口径与 y31/z2 完全一致 |
| 2~5. 像不像真人 ×4 | 一段 20s，盲听，三档绝对判断 | 定位：开头/2:30/5:00/7:20 四个位置 |
| 6. 像不像真人 ×1 | 同款题，**偷偷放 EL 段** | 隐藏对照：校准员工的标尺 |
| 7. 控制题 | 同一段放两遍 | 答题有效性（连续 4 轮全对） |

三档选项是**具体等级**不是兜底：①像真人 ②基本像但有机器感 ③机器感明显。
拿到答案改什么：
- 某段 ③ → 切那段波形做频谱/包络分析（定位精度 20s）
- 开头 vs 结尾档位差 → 漂移问题（分窗 F0 / 平坦度逐窗）
- EL 隐藏段也被打 ③ → 员工标尺过严，先校准再下结论
- 基线选我们 → 真进展（此前三轮 13 票全 EL）

素材：OURS = h5 线上成品（longprobe-z32-6cdd-1.mp3，494s）；
     EL = 案例库 6cdd25bc75f2/merged.wav（509s，同文案）。
"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
TMPD = "/mnt/tts/diag/_z35"
TGT_LUFS = -26.1
SR_CLIP = 24000

OURS = "/mnt/tts/output/longprobe-z32-6cdd/longprobe-z32-6cdd-1.mp3"
EL_S = "/mnt/tts/data/cases/6cdd25bc75f2/merged.wav"

SUFFIX = "20261016z4"

# (名字, 源文件, 起点, 时长, 说明)  —— rate 题 5 段；第 5 段是 EL 隐藏对照
RATE_SEGS = [
    ("z35_rate1", OURS, 0.0,  20.0, "我们的 h5 成品 开头 0:00-0:20"),
    ("z35_rate2", OURS, 150.0, 20.0, "我们的 h5 成品 2:30-2:50"),
    ("z35_rate3", OURS, 300.0, 20.0, "我们的 h5 成品 5:00-5:20"),
    ("z35_rate4", OURS, 440.0, 20.0, "我们的 h5 成品 7:20-7:40（近结尾）"),
    ("z35_rate5", EL_S, 240.0, 20.0, "🔴 EL 隐藏对照 4:00-4:20（员工不知道）"),
]
CTRL_SRC = (OURS, 60.0, 10.0)   # 控制题素材：成品 1:00-1:10，放两遍


def lufs(p):
    r = subprocess.run(["ffmpeg", "-i", p, "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    v = [l for l in r.stderr.splitlines() if "I:" in l and "LUFS" in l]
    return float(v[-1].split("I:")[1].split("LUFS")[0].strip()) if v else None


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()[:8]


def clip(src, s0, dur):
    t = os.path.join(TMPD, "_c.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-ss", "%.1f" % s0, "-ar", str(SR_CLIP), "-ac", "1",
                    "-t", "%.1f" % dur, t], check=True)
    got = os.path.getsize(t) / 2.0 / SR_CLIP
    if got < dur - 0.5:
        sys.exit("切片越界：%s 第%.0fs 起取%.0fs，实际仅%.1fs"
                 % (os.path.basename(src), s0, dur, got))
    return t


def emit(wav, name, title):
    """volume=%gdB 不是 %gdBB（多一个 B = +18dB）。"""
    g = TGT_LUFS - lufs(wav)
    out = os.path.join(FB_DIR, "%s.mp3" % name)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav,
                    "-af", "volume=%gdB" % g, "-b:a", "192k",
                    "-metadata", "title=%s" % title, out], check=True)
    back = lufs(out)
    print("  %-14s md5=%s  LUFS %6.1f %s"
          % (name, md5(out), back if back else 0,
             "OK" if back and abs(back - TGT_LUFS) < 1.0 else "!!爆表"))
    return out


def rate_item(iid, fid, note):
    """一段一题：音频放 ref 块（z 轮教训：放选项里会被当成选项①的音频），
    三个具体等级选项，盲听（不写我们的/EL/时间位置）。"""
    return {
        "id": iid, "dim": "像不像真人（绝对判断）",
        "title": "单独听这一段 20 秒：像不像是真人念的？",
        "desc": "请听下面这段 20 秒的音频（可以多听几遍），凭直觉选一个：\n\n"
                "① 像真人念的，没觉得有机器感\n"
                "② 基本像真人，但能听出一点机器感\n"
                "③ 机器感 / 电子音很明显",
        "ref": {"label": "点击播放这段音频", "file": fid, "note": ""},
        "note": note,
        "options": [
            {"key": "A", "file": None, "label": "① 像真人念的，没觉得有机器感"},
            {"key": "B", "file": None, "label": "② 基本像真人，但能听出一点机器感"},
            {"key": "C", "file": None, "label": "③ 机器感 / 电子音很明显"},
        ],
    }


def main():
    for p in (OURS, EL_S):
        if not os.path.exists(p):
            print("缺素材：%s" % p)
            return 1
    os.makedirs(TMPD, exist_ok=True)
    made = {}

    print("=== 切片 + 归一 ===")
    for nm, src, s0, dur, desc in RATE_SEGS:
        made[nm] = emit(clip(src, s0, dur), nm, "%s" % nm)
    made["z35_send_a"] = emit(clip(OURS, 0.0, 30.0), "z35_send_a", "send A")
    made["z35_send_b"] = emit(clip(EL_S, 0.0, 30.0), "z35_send_b", "send B")
    csrc, cs0, cdur = CTRL_SRC
    made["z35_ctrl1"] = emit(clip(csrc, cs0, cdur), "z35_ctrl1", "ctrl A")
    made["z35_ctrl2"] = emit(clip(csrc, cs0, cdur), "z35_ctrl2", "ctrl B")

    print("\n=== md5 自检 ===")
    inv = {}
    for nm, p in made.items():
        inv.setdefault(md5(p), []).append(nm)
    bad = False
    for h, ns in inv.items():
        if len(ns) > 1:
            print("  %-9s 撞车 -> %s" % (h, ns))
            bad = True
    if not bad:
        print("  无撞车")

    items = [
        {"id": "z4_send_baseline_%s" % SUFFIX, "dim": "发客户基线（长期锚点）",
         "title": "这 30 秒，哪一段可以直接发给客户？",
         "desc": "两段音频各 30 秒，听完选一个。\n\n"
                 "① 第 1 段可以\n② 第 2 段可以\n③ 都不行",
         "ref": None,
         "note": "锚点题：口径与 y31/z2 完全一致。A=我们 h5 成品开头 30s，"
                 "B=EL 同文案开头 30s。此前三轮 13 票全 B。"
                 "哪天出现 A/C 显著 → 真进展。",
         "options": [
             {"key": "A", "file": "z35_send_a.mp3", "label": "① 第 1 段（我们的）"},
             {"key": "B", "file": "z35_send_b.mp3", "label": "② 第 2 段（EL）"},
             {"key": "C", "file": None, "label": "③ 都不行"}]},
    ]
    for i, (nm, src, s0, dur, desc) in enumerate(RATE_SEGS, 1):
        fid = "%s.mp3" % nm
        items.append(rate_item(
            "z4_rate%d_%s" % (i, SUFFIX), fid,
            ("素材：%s。\n拿到答案改什么：选③ → 切该段 20s 波形做频谱/包络/"
             "分窗 F0 分析（定位精度 20 秒，不够下轮在这 20 秒里再切）；"
             "选① → 该段干净。四段档位若开头/结尾不一致 → 漂移方向。\n%s"
             % (desc,
                "🔴 本段是 EL 隐藏对照：员工也给③ → 标尺过严或问题非我们特有，"
                "先校准；EL=① 且我们=②③ → 差距确认。"
                if i == 5 else
                "（员工盲听，不知道是我们的还是 EL 的。）"))))
    items.append({
        "id": "z4_ctrl_same_%s" % SUFFIX, "dim": "答题有效性校验",
        "title": "【最后一题】这两段，听起来有区别吗？",
        "desc": "**这道题必须选一个**（选错也没关系）。\n\n"
                "这两段是**同一段音频放了两遍**，内容完全一样。听 5 秒就够。",
        "ref": None,
        "note": "隐藏控制题：正解=C 两段一样。问法沿用「这两段听起来有区别吗」"
                "——连续多轮全对；旧的「哪段念得更快」4 人里 2 人错，别再用。",
        "options": [
            {"key": "A", "file": "z35_ctrl1.mp3", "label": "A 明显有区别"},
            {"key": "B", "file": "z35_ctrl2.mp3", "label": "B 明显有区别"},
            {"key": "C", "file": None, "label": "C 没有区别，是一样的"}]})

    print("\n=== 题目自检 ===")
    ok = True
    fids = set(made.keys())
    for it in items:
        n_opt_audio = sum(1 for o in it["options"] if o.get("file"))
        r = (it.get("ref") or {}).get("file")
        # 所有引用的文件必须存在
        for o in it["options"]:
            if o.get("file") and not os.path.exists(os.path.join(FB_DIR, o["file"])):
                print("  %s 缺文件 %s" % (it["id"], o["file"]))
                ok = False
        if r:
            if not os.path.exists(os.path.join(FB_DIR, r)):
                print("  %s 缺 ref 文件 %s" % (it["id"], r))
                ok = False
            if n_opt_audio != 0:
                print("  %s ref 题不应再有选项音频" % it["id"])
                ok = False
        if n_opt_audio == 0 and not r:
            print("  %s 既无选项音频也无 ref -> 废题" % it["id"])
            ok = False
        if n_opt_audio == 1:
            print("  %s 单音频题必须改用 ref" % it["id"])
            ok = False
        if it["id"].startswith("z4_rate"):
            if len(it["options"]) != 3:
                print("  %s 三档选项数量不对" % it["id"])
                ok = False
            # 三档必须都是具体等级（无兜底）
            for o in it["options"]:
                if "说不上" in o["label"] or "不知道" in o["label"] or "其他" in o["label"]:
                    print("  %s 出现兜底选项：%s" % (it["id"], o["label"]))
                    ok = False
            # 盲听检查：题目文案不许出现 我们的/EL/段号
            blob = it["title"] + it["desc"]
            for kw in ("我们的", "EL", "第 1 段", "第1段", "结尾", "开头"):
                if kw in blob:
                    print("  %s 盲听题文案泄漏身份：%s" % (it["id"], kw))
                    ok = False
    n_rate = sum(1 for it in items if it["id"].startswith("z4_rate"))
    if n_rate != 5:
        print("  rate 题数 %d != 5（第 5 题必须是 EL 隐藏对照）" % n_rate)
        ok = False
    if ok:
        print("  题目结构 OK（%d 题：基线1 + 盲听评级5(含EL对照1) + 控制1）"
              % len(items))

    out = os.path.join(FB_DIR, "items.json")
    arc = os.path.join(FB_DIR, "items_roundz3.json")
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
