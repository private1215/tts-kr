#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第三十八轮题库：新版（块长 300）**直接对上 EL**，三个位置各一题

## 上一轮的结论（z5 裁决，3 人全有效）
- 中段 30s：2 票选新版 / 1 票听不出区别 / **0 票选旧版**
- **尾部 20s：3/3 全票选新版**（就是上次 3/3 判"明显电子"的那个位置）
→ 块长 600 → **300 确认留守**。

## 本轮又试了 150，长篇实测反而崩了（已回退 300）
| 指标 | z36(300) | z37(150) | EL |
|---|---|---|---|
| 20s 窗音色均值 | 0.781 | 0.802 | 0.934 |
| **20s 窗音色极差** | **0.184** | 0.396 🔴 | 0.042 |
| Σz² | 68.6 | 112.9 🔴 | 中位 59.6 |
| 平坦度 | +0.95σ | **+5.76σ** 🔴 | 0 |
| 硬切 | 1 | 9 🔴 | 2 |
🔴 **短篇（1200 字 raw）选出的最优块长 150，搬到长篇反而最差**
   → 又一次印证「短片不能当验收依据」，块长 300 是甜点，别再往下压。

## 这一轮问什么
块长这条线已收敛，接下来要知道**我们 vs EL 还差多少、差在哪个位置**：
三个位置各出一题「哪段更像真人在自然说话」+ 标准发客户基线（开头）+ 控制题。
拿到答案改什么：
- 尾部仍明显输 → 尾部还有专项问题（漂移/韵律），继续查
- 中段最接近 → 中段已达标，改进空间集中在首尾
- 三段都输 → 全局问题 = 引擎层（转路径 B 微调）
- 出现我们赢的位置 → 记录为首次突破
"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
TMPD = "/mnt/tts/diag/_z38"
TGT_LUFS = -26.1
SR_CLIP = 24000

OURS = "/mnt/tts/output/longprobe-z36-6cdd/longprobe-z36-6cdd-1.mp3"   # 块 300（现役）
EL_S = "/mnt/tts/data/cases/6cdd25bc75f2/merged.wav"

SUFFIX = "20261016z6"

# (文件名, 源, 起点, 时长, 位置说明)
CLIPS = [
    ("z38_head_a", OURS, 0.0, 30.0, "开头"),
    ("z38_head_b", EL_S, 0.0, 30.0, "开头"),
    ("z38_mid_a", OURS, 240.0, 30.0, "中段"),
    ("z38_mid_b", EL_S, 245.0, 30.0, "中段"),
    ("z38_tail_a", OURS, 440.0, 20.0, "尾部"),
    ("z38_tail_b", EL_S, 448.0, 20.0, "尾部"),
]


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


def like_item(iid, where, dur, a, b, note):
    """「哪段更像真人在自然说话」二选一。"""
    return {
        "id": iid, "dim": "像不像真人（%s）" % where,
        "title": "成品%s这两段（各 %d 秒），哪一段更像「真人在自然说话」？" % (where, dur),
        "desc": "随便听几遍，凭第一感觉选。\n"
                "重点听：像不像一个真人一直在说话，有没有机器味。",
        "ref": None, "note": note,
        "options": [
            {"key": "A", "file": a, "label": "① 第 1 段更像真人"},
            {"key": "B", "file": b, "label": "② 第 2 段更像真人"},
            {"key": "C", "file": None, "label": "③ 两段差不多，听不出区别"},
            {"key": "D", "file": None, "label": "④ 两段都像机器，都不像真人"}]}


def main():
    for p in (OURS, EL_S):
        if not os.path.exists(p):
            print("缺素材：%s" % p)
            return 1
    os.makedirs(TMPD, exist_ok=True)
    made = {}
    print("=== 切片 + 归一（A=我们的新版，B=EL）===")
    for nm, src, s0, dur, _w in CLIPS:
        made[nm] = emit(clip(src, s0, dur), nm, nm)
    made["z38_ctrl1"] = emit(clip(OURS, 120.0, 10.0), "z38_ctrl1", "ctrl A")
    made["z38_ctrl2"] = emit(clip(OURS, 120.0, 10.0), "z38_ctrl2", "ctrl B")

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
        {"id": "z6_send_baseline_%s" % SUFFIX, "dim": "发客户基线（长期锚点）",
         "title": "这 30 秒，哪一段可以直接发给客户？",
         "desc": "两段音频各 30 秒，听完选一个。\n\n"
                 "① 第 1 段可以\n② 第 2 段可以\n③ 都不行",
         "ref": None,
         "note": "锚点题：口径与 y31/z2/z4/z5 完全一致。A=我们的新版（块300）开头 30s，"
                 "B=EL 开头 30s。此前累计 19 票全 B。哪天出现 A → 首次突破。",
         "options": [
             {"key": "A", "file": "z38_head_a.mp3", "label": "① 第 1 段（我们的）"},
             {"key": "B", "file": "z38_head_b.mp3", "label": "② 第 2 段（EL）"},
             {"key": "C", "file": None, "label": "③ 都不行"}]},
        like_item("z6_mid_vs_el_%s" % SUFFIX, "中段", 30,
                  "z38_mid_a.mp3", "z38_mid_b.mp3",
                  "中段（约 4 分处）我们 vs EL。\n"
                  "选② → 中段仍有明显差距；选① → 中段已追上（记录首次突破）；\n"
                  "选③ → 中段差距已消失，问题集中在别的位置；选④ → 中段两边都不行。"),
        like_item("z6_tail_vs_el_%s" % SUFFIX, "尾部", 20,
                  "z38_tail_a.mp3", "z38_tail_b.mp3",
                  "**尾部（约 7 分 20 秒处）** 我们 vs EL —— 这是上一轮 3/3 全票"
                  "判「明显电子」的位置，块长改 300 后员工说改善最大，\n"
                  "所以这里要确认：**改善了，但离 EL 还有多远**。\n"
                  "选② → 尾部仍是最大短板（继续查漂移/韵律）；选③ → 尾部已追平 EL。"),
        {"id": "z6_ctrl_same_%s" % SUFFIX, "dim": "答题有效性校验",
         "title": "【最后一题】这两段，听起来有区别吗？",
         "desc": "**这道题必须选一个**（选错也没关系）。\n\n"
                 "这两段是**同一段音频放了两遍**，内容完全一样。听 5 秒就够。",
         "ref": None,
         "note": "隐藏控制题：正解=C 两段一样。",
         "options": [
             {"key": "A", "file": "z38_ctrl1.mp3", "label": "A 明显有区别"},
             {"key": "B", "file": "z38_ctrl2.mp3", "label": "B 明显有区别"},
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
    if ok:
        print("  题目结构 OK（基线1 + 中段1 + 尾部1 + 控制1）")

    out = os.path.join(FB_DIR, "items.json")
    arc = os.path.join(FB_DIR, "items_roundz5.json")
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
