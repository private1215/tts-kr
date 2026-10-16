#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第45轮（z13）裁决：**GS 微调 SOLA 女声 能不能顶替现役 omnivoice**？

## 本轮依据（engine_score_all.py，同文案全篇，case 00b6a45ad6aa）
| | 音色 | 覆盖率 | CER | 音高中位 | 句内动态 | 能量动态 | 句尾STD | 句尾降幅 | 语速 | 停顿/分 |
|---|---|---|---|---|---|---|---|---|---|---|
| EL 原声 | 1.000 | 0.954 | 0.045 | 265.8 | 9.55 | 15.33 | 6.90 | +0.55 | 5.35 | 20.0 |
| 现役 omnivoice | 0.858 | 0.963 | 0.038 | 267.9 | 9.12 | 14.28 | 5.83 | −0.72 | 5.35 | 23.7 |
| **GS 微调 SOLA** | **0.908** | 0.928 | 0.070 | 274.0 | 8.22 | 15.91 | 4.66 | −2.58 | 5.46 | 24.6 |

判定：现役「音色偏低(0.8582)」，**GS 微调「通过」**。
→ GS 音色 +0.050（女声缺口 0.177 里补掉近 1/3），且**没有翻车**
  （覆盖率 0.928 ≥ 0.85；历史上 GS 那次翻车覆盖率只有 0.150）。
→ 但弱点也清楚：句内动态 8.22（现役 9.12 / EL 9.55）偏小、
  句尾降幅 −2.58（现役 −0.72）下沉更多 —— 可能正好是员工说的
  「女生的少了点感情」。**所以必须让员工盲听裁决，不能只看数字。**

## 公平性
两条都走 `POST /api/jobs` 的**同一条流水线**（分块→引擎→ASR回检→PBA→
慢速修复→EQ→PROSO→呼吸→响度闭环→peakfit），只有"谁来念"不同。
同一条文案、同一个 case_id、同一个 seed=42。

## 🔴 切点必须**按文本对齐**，不能按绝对秒
三条时长不同（EL 459.13 / 现役 458.83 / **GS 450.38**），
GS 比 EL 快 8.75s → 同一个绝对秒根本不是同一句话，
员工会变成在比"哪句好念"（正是用户反复纠正的陷阱）。
`align47.py` 用 ASR 实测三方同一句话的相似度，本轮只取
**三方都 > 0.85** 的点：t=12(0.975/0.957) / t=115(0.902/0.911) / t=320(0.907/0.943)；
隐藏对照用 t=30(0.949/0.947)。

## 盲听
选项只写「① 第 1 段 / ② 第 2 段 / ③ 差不多 / ④ 都不行」，
**不许出现「我们的 / EL / 标杆 / 参考 / 标准」**；身份只在 note 里记。

用法：/mnt/tts/projects/qwen3-tts/venv/bin/python mk_z45.py
"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
TMPD = "/mnt/tts/diag/_z45"
TGT_LUFS = -26.1
SR_CLIP = 24000
DUR = 20.0

EL = "/mnt/tts/data/cases/00b6a45ad6aa/merged.wav"      # 459.13
CUR = "/mnt/tts/diag/sola47_f_omnivoice.wav"            # 458.83  现役
GS = "/mnt/tts/diag/sola47_f_gptsovits.wav"             # 450.38  GS微调

# align47.py 实测：EL 时刻 -> 各自的换算系数
SC_CUR = 458.83 / 459.13
SC_GS = 450.38 / 459.13

# (素材名, 源文件, EL 时刻)
CLIPS = [
    ("z45_a_s1", CUR, 12.0), ("z45_b_s1", GS, 12.0),
    ("z45_a_s2", CUR, 115.0), ("z45_b_s2", GS, 115.0),
    ("z45_a_s3", CUR, 320.0), ("z45_b_s3", GS, 320.0),
    # 隐藏对照：A=现役(我们)  B=EL 同秒同句
    ("z45_a_el", CUR, 30.0), ("z45_b_el", EL, 30.0),
]
SUFFIX = "20261016z13"


def scale(src, t):
    if src == CUR:
        return t * SC_CUR
    if src == GS:
        return t * SC_GS
    return t


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
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ss", "%.2f" % s0,
                    "-ar", str(SR_CLIP), "-ac", "1", "-t", "%.1f" % dur, t],
                   check=True)
    got = os.path.getsize(t) / 2.0 / SR_CLIP
    if got < dur - 0.5:
        sys.exit("切片越界：%s 第%.1fs 起，实际%.1fs" % (src, s0, got))
    return t


def emit(wav, name):
    g = TGT_LUFS - lufs(wav)
    out = os.path.join(FB_DIR, "%s.mp3" % name)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav,
                    "-af", "volume=%gdB" % g, "-b:a", "192k",
                    "-metadata", "title=%s" % name, out], check=True)
    back = lufs(out)
    print("  %-11s 起%.1fs  md5=%s  LUFS %6.1f %s"
          % (name, 0.0, md5(out), back if back else 0,
             "OK" if back and abs(back - TGT_LUFS) < 1.0 else "!!爆表"))
    return out


def two(iid, title, desc, a, b, note):
    return {
        "id": iid, "dim": "女声·整体", "title": title, "desc": desc,
        "ref": None, "note": note,
        "options": [
            {"key": "A", "file": a, "label": "① 第 1 段更像真人"},
            {"key": "B", "file": b, "label": "② 第 2 段更像真人"},
            {"key": "C", "file": None, "label": "③ 两段差不多，听不出区别"},
            {"key": "D", "file": None, "label": "④ 两段都不像真人"}]}


def main():
    for p in (EL, CUR, GS):
        if not os.path.exists(p):
            print("缺素材：%s" % p)
            return 1
    os.makedirs(TMPD, exist_ok=True)
    made = {}
    print("=== 切片 + 归一（20s × 8，切点按 align47 文本对齐）===")
    for nm, src, t in CLIPS:
        made[nm] = emit(clip(src, scale(src, t), DUR), nm)
    made["z45_ctrl1"] = emit(clip(CUR, 200.0, 6.0), "z45_ctrl1")
    made["z45_ctrl2"] = emit(clip(CUR, 200.0, 6.0), "z45_ctrl2")

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

    D = ("两段读的是**同一段话**，用不同的方式生成。\n"
         "**只问一件事：哪一段听起来更像真人在说话？**\n"
         "听一遍就选，凭第一感觉，不用分析原因。")

    items = [
        two("z13_s1_%s" % SUFFIX,
            "【第 1 段】这两段，哪一段更像真人？", D,
            "z45_a_s1.mp3", "z45_b_s1.mp3",
            "裁决①：现役 omnivoice(①) vs GS 微调 SOLA(②)，切点 = 全文第 12s\n"
            "（align47 实测三方同句相似度：现役/EL 0.975、GS/EL 0.957 —— 本轮最齐的一点）。\n"
            "客观：音色 0.858 → 0.908（+0.050）；覆盖率 0.963 → 0.928；CER 0.038 → 0.070。\n"
            "选② → GS 微调方向成立，可考虑切引擎；选①/③ → 维持现役。"),
        two("z13_s2_%s" % SUFFIX,
            "【第 2 段】这两段，哪一段更像真人？", D,
            "z45_a_s2.mp3", "z45_b_s2.mp3",
            "裁决②：切点 = 全文第 115s（相似度 现役/EL 0.902、GS/EL 0.911，GS 略更齐）。\n"
            "中段换一个位置复验，避免「只在开头成立」。"),
        two("z13_s3_%s" % SUFFIX,
            "【第 3 段】这两段，哪一段更像真人？", D,
            "z45_a_s3.mp3", "z45_b_s3.mp3",
            "裁决③：切点 = 全文第 320s（相似度 现役/EL 0.907、GS/EL 0.943）。\n"
            "🔴 后段是 GPT-SoVITS 已知弱项（长文本发散/吞字），必须单独验一次。\n"
            "若这里明显崩，哪怕前两题全赢也不能切引擎。"),
        two("z13_el_%s" % SUFFIX,
            "【第 4 段】这两段，哪一段更像真人？", D,
            "z45_a_el.mp3", "z45_b_el.mp3",
            "🔴 **隐藏对照**：① = 现役（我们），② = EL 原声，切点 = 全文第 30s\n"
            "（相似度 现役/EL 0.949）。员工不知道身份。\n"
            "这是「能否直接发客户」基线题的盲听版：历史上 13 票全判 EL。\n"
            "本轮如果 ② 仍然明显占优 → 说明题目有效、员工判别力在线，\n"
            "第 1~3 题的票才有含金量；若第 1~3 题里 ① 赢了而这里 ② 赢，\n"
            "说明**换引擎确实拉近了差距**。"),
        {"id": "z13_ctrl_same_%s" % SUFFIX, "dim": "答题有效性校验",
         "title": "【最后一题】这两段，听起来有区别吗？",
         "desc": "**这道题必须选一个**（选错也没关系）。\n\n"
                 "这两段是**同一段音频放了两遍**，内容完全一样。听 5 秒就够。",
         "ref": None, "note": "隐藏控制题：正解 = C 两段一样。答错整份作废。",
         "options": [
             {"key": "A", "file": "z45_ctrl1.mp3", "label": "A 明显有区别"},
             {"key": "B", "file": "z45_ctrl2.mp3", "label": "B 明显有区别"},
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
        if n == 0:
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
        print("  题目结构 OK（裁决×3 + 隐藏 EL 对照×1 + 控制×1），全盲听")

    out = os.path.join(FB_DIR, "items.json")
    arc = os.path.join(FB_DIR, "items_roundz12.json")
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
