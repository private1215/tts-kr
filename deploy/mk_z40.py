#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第四十轮题库：女声 EQ **换档位复听** + 新增「吃字」专项裁决

## 为什么要换档位
z7 用了 highshelf 4000 −2dB，客观很好（Σz² 32.2→18.9、4k~8k +2.13σ→−1.45σ、
音色不变），但员工反馈：

> 「第二题 B 选项 —— 卡字、吃字，中间几个字」（디아 自由文本："第二题断句有异声"）

🔴 **先排除伪信号**：A/B 是同一段音频（女声成品 90~120s），只差一条 EQ。
   `stuck_check.py` 实测：**重复(卡字)=0 处、硬切=0、最陡落差 27dB**，
   且**碎停顿结构和 EL 一模一样**（我们 48 处 vs EL 45 处 / 男声 39 vs 42）
   → 没有客观卡字，EQ 也**没有**制造停顿或硬切。
   → 那"吃字"只可能来自 EQ 本身：4k~8k 少 2dB 把**齿音/塞音辅音**压轻了。

## 新档位：绕开辅音区（辅音能量主要在 2~6k）
| 档位 | Σz² | 平坦度 | 4k~8k |
|---|---|---|---|
| 现役 | 32.2 | +3.85σ | +2.13σ |
| HS4000 −1.5 | 19.1 | +3.06σ | −0.64σ |
| HS4000 −2（z7 用的） | 18.9 | +2.81σ | −1.45σ |
| HS6000 −3 | 18.5 | +2.64σ | −1.47σ |
| **HS8000 −4（本轮）** | **17.8** | +2.68σ | **−1.06σ** |

→ HS8000 −4：Σz² 最优、4k~8k 只动 −1.06σ、**几乎不碰 2~6k 辅音区**。

## 题（4 题，含一道"吃字"专项）
1. 中段：现役 vs 新档位，哪段更像真人
2. **中段：哪一段的"字更清楚"（针对吃字反馈）** ← 新增专项
3. 尾部：现役 vs 新档位，哪段更像真人
4. 控制题
🔴 全部盲听（选项不写我们的/EL，自检会拦）
"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
TMPD = "/mnt/tts/diag/_z40"
TGT_LUFS = -26.1
SR_CLIP = 24000

OURS = "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"   # 女声现役成品 198.7s
EQ_F = "highshelf=f=8000:g=-4"      # 本轮候选（绕开 2~6k 辅音区）

SUFFIX = "20261016z8"

CLIPS = [
    ("z40_mid_a", 90.0, 30.0, ""),      # 现役 中段
    ("z40_mid_b", 90.0, 30.0, EQ_F),    # 新档位 中段
    ("z40_tail_a", 165.0, 20.0, ""),    # 现役 尾部
    ("z40_tail_b", 165.0, 20.0, EQ_F),  # 新档位 尾部
]


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


def emit(wav, name, title):
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


def two(iid, title, desc, a, b, la, lb, note):
    """通用二选一裁决（盲听，4 选项：A / B / 差不多 / 都不行）。"""
    return {
        "id": iid, "dim": "女声", "title": title, "desc": desc,
        "ref": None, "note": note,
        "options": [
            {"key": "A", "file": a, "label": la},
            {"key": "B", "file": b, "label": lb},
            {"key": "C", "file": None, "label": "③ 两段差不多，听不出区别"},
            {"key": "D", "file": None, "label": "④ 两段都像机器，都不像真人"}]}


def main():
    if not os.path.exists(OURS):
        print("缺素材：%s" % OURS)
        return 1
    os.makedirs(TMPD, exist_ok=True)
    made = {}
    print("=== 切片 + 归一（女声，A=现役 / B=新档位 HS8000 -4）===")
    for nm, s0, dur, af in CLIPS:
        made[nm] = emit(clip(s0, dur, af), nm, nm)
    made["z40_ctrl1"] = emit(clip(60.0, 10.0), "z40_ctrl1", "ctrl A")
    made["z40_ctrl2"] = emit(clip(60.0, 10.0), "z40_ctrl2", "ctrl B")

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
        two("z8_f_mid_%s" % SUFFIX,
            "【女声·中段】这两段（各 30 秒），哪一段更像「真人在自然说话」？",
            "两段是同一版女声，第 2 段做了一点音质调整。凭第一感觉选。",
            "z40_mid_a.mp3", "z40_mid_b.mp3",
            "① 第 1 段更像真人", "② 第 2 段更像真人",
            "裁决：现役(①) vs HS8000 −4(②)。\n"
            "选② → 上线该档位（Σz² 32.2→17.8、平坦度 +3.85σ→+2.68σ）。\n"
            "选①/③ → 女声高频不是短板，别再动 EQ。"),
        two("z8_f_clear_%s" % SUFFIX,
            "【女声·吃字专项】这两段，哪一段的「字听得更清楚」？",
            "上一轮有同事反馈「卡字、吃字」，所以单独再问一次。\n"
            "重点听：字有没有被吃掉、含不含糊，不用管像不像真人。",
            "z40_mid_a.mp3", "z40_mid_b.mp3",
            "① 第 1 段字更清楚", "② 第 2 段字更清楚",
            "🔴 吃字专项（新题型）。背景：z7 用 HS4000 −2dB，员工反馈吃字；\n"
            "客观核查：重复=0、硬切=0、碎停顿与 EL 一致 → 不是真卡字，\n"
            "推测是 2~6k 辅音被压轻。本轮档位改成 8k 以上，绕开辅音区。\n"
            "选②/③ → 吃字消失，档位可用；选① → 新档位反而更糊，回退 HS4000 −1.5。"),
        two("z8_f_tail_%s" % SUFFIX,
            "【女声·尾部】这两段（各 20 秒），哪一段更像「真人在自然说话」？",
            "两段是同一版女声的结尾部分，第 2 段做了同样的调整。",
            "z40_tail_a.mp3", "z40_tail_b.mp3",
            "① 第 1 段更像真人", "② 第 2 段更像真人",
            "尾部复核：确认新档位在尾部也成立（男声的短板就在尾部）。"),
        {"id": "z8_ctrl_same_%s" % SUFFIX, "dim": "答题有效性校验",
         "title": "【最后一题】这两段，听起来有区别吗？",
         "desc": "**这道题必须选一个**（选错也没关系）。\n\n"
                 "这两段是**同一段音频放了两遍**，内容完全一样。听 5 秒就够。",
         "ref": None, "note": "隐藏控制题：正解=C 两段一样。",
         "options": [
             {"key": "A", "file": "z40_ctrl1.mp3", "label": "A 明显有区别"},
             {"key": "B", "file": "z40_ctrl2.mp3", "label": "B 明显有区别"},
             {"key": "C", "file": None, "label": "C 没有区别，是一样的"}]},
    ]

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
        if n >= 2:      # 🔴 盲听铁律
            blob = it["title"] + it["desc"] + "".join(
                o["label"] for o in it["options"])
            for kw in ("我们的", "EL", "标杆", "参考", "标准"):
                if kw in blob:
                    print("  %s 泄漏身份（盲听题）：%s" % (it["id"], kw))
                    ok = False
    if ok:
        print("  题目结构 OK（中段1 + 吃字专项1 + 尾部1 + 控制1）")

    out = os.path.join(FB_DIR, "items.json")
    arc = os.path.join(FB_DIR, "items_roundz7.json")
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
