#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第三十六轮题库：块长 600 vs 300 的**二选一裁决**（员工是最终裁判）

## 背景（本轮客观测量）
- 员工第35轮：尾部 7:20 段 3/3 全票「机器感很明显」，EL 隐藏对照 1.33 → 差距真实
- `drift_win.py`：我们长篇**逐窗音色极差 0.412**，EL 只有 **0.042**（10 倍）
- `steps_ab.py`：提 STEPS 无效（s30 极差 0.208 > s10 0.147）→ 已排除
- `chunk_ab.py`：块长 600→300→150，极差 0.153→0.096→0.061（块内音色衰减所致）
- 已上线 VOX_CHUNK_CHARS=300，长篇回归 `longprobe-z36-6cdd`：
  | 指标 | z32(600) | z36(300) | EL |
  |---|---|---|---|
  | 20s 窗音色均值 | 0.696 | **0.781** | 0.934 |
  | **20s 窗音色极差** | 0.412 | **0.184** | 0.042 |
  | Σz² | 19.6 | 68.6（仍在 EL 17~240 内）| 中位 59.6 |
  | 硬切 | 2 | 1 | 2 |
  | 语速 | 8.36 | 8.37 | 8.37 |

🔴 Σz² 19.6→68.6 是**变坏**（虽然落回 EL 分布内），所以不能我们自己拍板：
   **交员工二选一裁决**（此类题历史最稳，第26轮 3/3 一致）。
   「指标和听感冲突时以听感为准」。

## 题（4 题，每题只点一下）
1. 中段 30s：A=旧(块600) B=新(块300)，哪段更像真人在自然说话
2. **尾部 20s（上次 3/3 判"明显电子"的那段）**：A=旧 B=新，哪段更像真人
3. 发客户基线：我们的新版 vs EL（锚点口径不变，看是否首次出现变化）
4. 控制题
"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
TMPD = "/mnt/tts/diag/_z36"
TGT_LUFS = -26.1
SR_CLIP = 24000

OLD = "/mnt/tts/output/longprobe-z32-6cdd/longprobe-z32-6cdd-1.mp3"   # 块 600
NEW = "/mnt/tts/output/longprobe-z36-6cdd/longprobe-z36-6cdd-1.mp3"   # 块 300
EL_S = "/mnt/tts/data/cases/6cdd25bc75f2/merged.wav"

SUFFIX = "20261016z5"

# (文件名, 源, 起点, 时长)
CLIPS = [
    ("z36_mid_a", OLD, 240.0, 30.0),     # 中段 A = 旧
    ("z36_mid_b", NEW, 240.0, 30.0),     # 中段 B = 新
    ("z36_tail_a", OLD, 440.0, 20.0),    # 尾部 A = 旧（上次 3/3 判电子的那段位置）
    ("z36_tail_b", NEW, 440.0, 20.0),    # 尾部 B = 新
    ("z36_send_a", NEW, 0.0, 30.0),      # 基线：我们的新版
    ("z36_send_b", EL_S, 0.0, 30.0),     # 基线：EL
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


def ab_item(iid, title, desc, a, b, note):
    """二选一裁决：①A ②B ③差不多 ④都不行（④=两段都像机器，不是兜底）。"""
    return {
        "id": iid, "dim": "二选一裁决", "title": title, "desc": desc,
        "ref": None, "note": note,
        "options": [
            {"key": "A", "file": a, "label": "① 第 1 段更像真人在说话"},
            {"key": "B", "file": b, "label": "② 第 2 段更像真人在说话"},
            {"key": "C", "file": None, "label": "③ 两段差不多，听不出区别"},
            {"key": "D", "file": None, "label": "④ 两段都像机器，都不像真人"}]}


def main():
    for p in (OLD, NEW, EL_S):
        if not os.path.exists(p):
            print("缺素材：%s" % p)
            return 1
    os.makedirs(TMPD, exist_ok=True)
    made = {}
    print("=== 切片 + 归一 ===")
    for nm, src, s0, dur in CLIPS:
        made[nm] = emit(clip(src, s0, dur), nm, nm)
    made["z36_ctrl1"] = emit(clip(NEW, 120.0, 10.0), "z36_ctrl1", "ctrl A")
    made["z36_ctrl2"] = emit(clip(NEW, 120.0, 10.0), "z36_ctrl2", "ctrl B")

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
        ab_item("z5_mid_ab_%s" % SUFFIX,
                "这两段（各 30 秒），哪一段更像「真人在自然说话」？",
                "随便听几遍都行，凭第一感觉选。\n"
                "重点听：像不像一个真人一直在说话，有没有机器味。",
                "z36_mid_a.mp3", "z36_mid_b.mp3",
                "裁决：块长 600(①) vs 300(②)。\n"
                "选② → 300 更好听，确认上线（音色极差 0.412→0.184）；"
                "Σz² 68.6 的代价认了。\n"
                "选① → 回退 VOX_CHUNK_CHARS=600，另找路子。\n"
                "选③ → 员工听不出差别 → 保留 300（客观指标更好且不更难听）。\n"
                "选④ → 两段都电子 → 块长不是解药，回到引擎/微调。"),
        ab_item("z5_tail_ab_%s" % SUFFIX,
                "这两段（各 20 秒），哪一段更像「真人在自然说话」？",
                "这两段取自成品**靠结尾**的位置（约 7 分 20 秒处）——\n"
                "上一轮有同事说这个位置「机器感很明显」，所以单独再比一次。",
                "z36_tail_a.mp3", "z36_tail_b.mp3",
                "尾部专测（第35轮 3/3 全票判「明显电子」的位置）。\n"
                "若②明显赢 → 块长 300 修好了尾部衰减，这是最大收益点。\n"
                "若①赢或③ → 尾部问题另有原因（漂移/韵律），继续查。"),
        {"id": "z5_send_baseline_%s" % SUFFIX, "dim": "发客户基线（长期锚点）",
         "title": "这 30 秒，哪一段可以直接发给客户？",
         "desc": "两段音频各 30 秒，听完选一个。\n\n"
                 "① 第 1 段可以\n② 第 2 段可以\n③ 都不行",
         "ref": None,
         "note": "锚点题：口径与 y31/z2/z4 完全一致。A=我们的**新版（块300）**开头 30s，"
                 "B=EL 同文案开头 30s。此前四轮 16 票全 B。"
                 "哪天出现 A → 真进展。",
         "options": [
             {"key": "A", "file": "z36_send_a.mp3", "label": "① 第 1 段（我们的）"},
             {"key": "B", "file": "z36_send_b.mp3", "label": "② 第 2 段（EL）"},
             {"key": "C", "file": None, "label": "③ 都不行"}]},
        {"id": "z5_ctrl_same_%s" % SUFFIX, "dim": "答题有效性校验",
         "title": "【最后一题】这两段，听起来有区别吗？",
         "desc": "**这道题必须选一个**（选错也没关系）。\n\n"
                 "这两段是**同一段音频放了两遍**，内容完全一样。听 5 秒就够。",
         "ref": None,
         "note": "隐藏控制题：正解=C 两段一样。",
         "options": [
             {"key": "A", "file": "z36_ctrl1.mp3", "label": "A 明显有区别"},
             {"key": "B", "file": "z36_ctrl2.mp3", "label": "B 明显有区别"},
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
        if it["id"].startswith("z5_") and "ab" in it["id"]:
            if n != 2:
                print("  %s 裁决题应有 2 段音频，实为 %d" % (it["id"], n))
                ok = False
            if len(it["options"]) != 4:
                print("  %s 裁决题应有 4 个选项（含差不多/都不行）" % it["id"])
                ok = False
    if ok:
        print("  题目结构 OK（裁决2 + 基线1 + 控制1）")

    out = os.path.join(FB_DIR, "items.json")
    arc = os.path.join(FB_DIR, "items_roundz4.json")
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
