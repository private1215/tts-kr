#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第二十八轮（u）题库：**验证「卡字」是否被整篇1块解决**（闭环 ③④）

## 这一轮存在的唯一理由
第 27 轮闭环已完成 ①②③，④就是这一轮。

第 27 轮虽然只有 2 份有效答卷，但**作废答卷的文本给出了最关键的线索**：
24124112 写「17秒 有一个字 多出来**瞬间卡顿**」「**吃字，卡字了**」

🔴 已定位根因并做了受控实验（唯一变量 `TTS_CHUNK_CHARS` 2600→5000）：
| | 拼接点最陡下降 | 位置 |
|---|---|---|
| 2 块（旧） | **−157.3 dB** | 340.4s ← 数字真零→发声的**硬切** |
| **1 块（新，已上线）** | **−19.6 dB** | 347.8s ← 正常自然停顿 |

→ **"瞬间卡顿"的物理来源就是分块拼接点无过渡。**

## 本轮只问三件事，每件都对应一个动作
| 题 | 答案 → 动作 |
|---|---|
| u1 卡字/吃字还在吗 | 还在 → 分块不是根因，查引擎稳定性；没了 → 保留 1 块并查单块上限 |
| u2 语速稳不稳 | 不稳 → 查拼接/韵律；稳 → 排除（此前"고구마说忽快忽慢」）|
| u3 能否直接发客户 | 长期基线锚点（此前 15 票全选 EL）|

🔴 **不再问「电子感」**：第27轮 1 人"能接受"1 人"还是很合成"，
   两人相反 → 现阶段问不出信息量，**先修已证实的问题（卡字）再回头**。
   （这也是「不要为了反馈而反馈」：问不出动作的题不出。）

## 素材
全部取自真长文，且**都是同一份 6cdd 文案**：
- 新（1块，已上线）：/mnt/tts/output/longprobe-omni-1block/（553.6s）
- 旧（2块）：/mnt/tts/output/longprobe-omni-6cdd/（553.3s）
- EL 标杆：/mnt/tts/data/cases/6cdd25bc75f2/merged.wav（509s）
"""
import hashlib
import json
import os
import shutil
import subprocess

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
TMPD = "/mnt/tts/diag/_u28"
TGT_LUFS = -26.1
DUR = 30.0

NEW_ = "/mnt/tts/output/longprobe-omni-1block/longprobe-omni-1block-1.mp3"
OLD_ = "/mnt/tts/output/longprobe-omni-6cdd/longprobe-omni-6cdd-1.mp3"
EL_S = "/mnt/tts/data/cases/6cdd25bc75f2/merged.wav"

# 🔴 关键：**必须包含拼接点 340 秒** —— 24124112 就是在那附近听到卡顿
P_E, P_BOUND, P_L = 100.0, 330.0, 400.0


def lufs(p):
    r = subprocess.run(["ffmpeg", "-i", p, "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    v = [l for l in r.stderr.splitlines() if "I:" in l and "LUFS" in l]
    return float(v[-1].split("I:")[1].split("LUFS")[0].strip()) if v else None


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()[:8]


def cut(src, start, dur):
    t = os.path.join(TMPD, "_t%d.wav" % int(start * 1000))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-ss", "%.1f" % start, "-ar", "16000", "-ac", "1",
                    "-t", "%.1f" % dur, t], check=True)
    got = os.path.getsize(t) / 2.0 / 16000
    if got < dur - 0.5:
        raise SystemExit("🔴 切片越界：%s 第%.0fs起取%.0fs，实际仅%.1fs"
                         % (os.path.basename(src), start, dur, got))
    return t


def emit(wav, name, title):
    g = TGT_LUFS - lufs(wav)
    mp3 = os.path.join(FB_DIR, "%s.mp3" % name)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav,
                    "-af", "volume=%gdB" % g,       # 🔴 dB 不是 %gdBB
                    "-b:a", "192k", "-metadata", "title=%s" % title, mp3],
                   check=True)
    back = lufs(mp3)
    if back is None:
        print("  %-12s 🔴 读不出 LUFS" % name)
        return None
    print("  %-12s md5=%s  LUFS %6.1f %s"
          % (name, md5(mp3), back,
             "OK" if abs(back - TGT_LUFS) < 1.0 else "!!爆表"))
    return mp3


def main():
    os.makedirs(TMPD, exist_ok=True)
    made = {}
    for tag, src in (("new", NEW_), ("old", OLD_), ("el", EL_S)):
        for pos, st in (("e", P_E), ("b", P_BOUND), ("l", P_L)):
            nm = "u28_%s_%s" % (tag, pos)
            made[nm] = emit(cut(src, st, DUR), nm, "clip %s" % nm)

    ctrl = cut(NEW_, 200.0, DUR)
    made["u28_ctrl1"] = emit(ctrl, "u28_ctrl1", "ctrl A")
    made["u28_ctrl2"] = emit(ctrl, "u28_ctrl2", "ctrl B")

    print("\n=== md5 自检 ===")
    inv = {}
    for nm, p in made.items():
        inv.setdefault(md5(p), []).append(nm)
    bad = False
    for h, ns in inv.items():
        if len(ns) > 1:
            ok = set(ns) <= {"u28_ctrl1", "u28_ctrl2"}
            print("  %-9s %s" % (h, "OK(控制题同素材)" if ok
                                 else "🔴 撞车 -> %s" % ns))
            if not ok:
                bad = True
    if not bad:
        print("  ✅ 无撞车")

    items = [
        # ---- 1. 核心：拼接点还在不在（直接验证这次修复）----
        {"id": "u1_stuck_check_20261015u", "dim": "拼接点验证",
         "title": "这两段，哪一段有「突然卡一下 / 吃字」的地方？",
         "desc": "两段都是男声，同一段文案，各 30 秒（**都在第 5 分半附近**）。\n\n"
                 "**这一题请仔细听有没有哪个字/哪个瞬间是突然卡住的。**\n"
                 "如果两段都听不出卡顿，就选最后一项。",
         "ref": None,
         "note": "🔴🔴 本轮核心题，素材位置就是**拼接点 340s 附近**。\n"
                 "第1段=整篇1块（现在线上的版本）\n"
                 "第2段=分2块（旧版本，拼接点在这里）\n\n"
                 "机器侧已证实：旧版在 340.4s 有 **−157.3dB 的硬跳变**\n"
                 "（数字真零直接跳到发声），新版同位置只有 −19.6dB（正常自然停顿）。\n\n"
                 "→ 这题问**耳朵能不能听出这个硬切**。\n"
                 "选C（都听不出）→ 修复有效，且说明以前听错的地方；"
                 "选B（旧版卡）→ 修复被证实。",
         "options": [
             {"key": "A", "file": "u28_new_b.mp3",
              "label": "第 1 段（现在的版本）有卡顿"},
             {"key": "B", "file": "u28_old_b.mp3",
              "label": "第 2 段（旧版本）有卡顿"},
             {"key": "C", "file": None, "label": "两段都听不出卡顿"}]},

        # ---- 2. 语速稳不稳（고구마 说「忽快忽慢」）----
        {"id": "u2_speed_stable_20261015u", "dim": "语速稳定性",
         "title": "这段从头听下来，语速稳不稳？",
         "desc": "一段男声，30 秒，是**现在线上的版本**。\n\n"
                 "就判断语速有没有忽快忽慢。",
         "ref": {"label": "▶ 点这个播放", "file": "u28_new_l.mp3",
                 "note": "现在线上的版本。"},
         "note": "고구마 第27轮写「**语速忽快忽慢**」，"
                 "但该答卷控制题答 B 已作废 → **这条线索待复验**。\n"
                 "若此处也说不稳 → 语速是下一个要修的点（拼接/韵律层）。",
         "options": [
             {"key": "A", "file": None, "label": "① 稳，全程一个速度"},
             {"key": "B", "file": None, "label": "② 忽快忽慢，有明显变速"},
             {"key": "C", "file": None, "label": "③ 说不上来"}]},

        # ---- 3. 长期基线锚点 ----
        {"id": "u3_send_baseline_20261015u", "dim": "可发送性",
         "title": "这 30 秒，哪一段可以直接发给客户？",
         "desc": "两段都是男声，同一段文案，各 30 秒。\n\n"
                 "按「能不能直接发给客户」判断。",
         "ref": None,
         "note": "**长期基线锚点。**\n"
                 "此前 4 轮共 **15 张票无一例外全选 EL**。\n"
                 "→ 出现「我们也能发」才算真有进展。",
         "options": [
             {"key": "A", "file": "u28_new_e.mp3", "label": "第 1 段（我们的）"},
             {"key": "B", "file": "u28_el_e.mp3", "label": "第 2 段（EL）"},
             {"key": "C", "file": None, "label": "都不行"}]},
    ]

    # 控制题改问法：第27轮 4 人里 2 人答错 → 「念得更快」可能太抽象
    items.append(
        {"id": "ctrl_same_20261015u", "dim": "答题有效性校验",
         "title": "【最后一题】这两段，听起来有区别吗？",
         "desc": "**这道题必须选一个**（选错也没关系）。\n\n"
                 "这两段是**同一段音频放了两遍**，内容完全一样。\n"
                 "听 5 秒就够。",
         "ref": None,
         "note": "隐藏控制题：正解=C 两段一样。\n"
                 "🔴 换问法的原因：第27轮用「哪段念得更快」，4 人里 2 人答错\n"
                 "（答A或B）。可能「更快」太抽象或被猜出来。\n"
                 "改问「有没有区别」更直白，正解仍是「一样」。",
         "options": [
             {"key": "A", "file": "u28_ctrl1.mp3", "label": "第 1 段不一样"},
             {"key": "B", "file": "u28_ctrl2.mp3", "label": "第 2 段不一样"},
             {"key": "C", "file": None, "label": "两段一样，听不出区别"}]})

    print("\n=== 题目完整性 ===")
    empty = []
    for it in items:
        hf = any(o.get("file") for o in it.get("options") or [])
        hr = bool(it.get("ref") and it["ref"].get("file"))
        if not (hf or hr):
            empty.append(it["id"])
        print("  %-30s 选项=%s 参考=%s"
              % (it["id"], "有" if hf else "无", "有" if hr else "无"))
    print("  ✅ 每题都有音频" if not empty else "  🔴 废题: %s" % empty)

    p = os.path.join(FB_DIR, "items.json")
    if os.path.exists(p):
        shutil.copyfile(p, p + ".bak-before-u")
        print("\n已备份旧题库 ->", p + ".bak-before-u")
    json.dump({"items": items}, open(p, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("题库已写入 %s  共 %d 题" % (p, len(items)))
    if bad or empty:
        print("🔴 存在拦截项，先别让员工答题")
    print("\n发布后确认：curl -s http://服务器:8000/api/fb/where")
    print("归档：python3 tools/fb_loop.py --archive u")


if __name__ == "__main__":
    main()