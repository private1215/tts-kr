#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第三十四轮题库：**把音频切好，让员工只做点选**（零技能版）

## 为什么废掉上一轮（mk_z33.py 的定点标注）
上一轮用了前端 `mode: diagnostic`：要员工「听到问题→暂停→点按钮→再打字」。
🔴 这是**比反应速度**，而且一慌就点晚，秒数是错的——
   错误的秒数比没有秒数更糟。员工是非专业人士，不能用这个。

## 这一轮改成「分段点选」
把同一段 90 秒切成 N 段，每段一个独立播放器，让员工**只点一下**
「哪一段最明显」。不需要暂停、不需要计时、不需要打字。

- 前端是单选（`feedback.html:225 PICK[id] = o.dataset.k`），所以问「最明显的是哪一段」
- 10 秒一段 → 一次定位精度 10 秒；不满意下轮再在那 10 秒里切 10 段，收敛很快
- 必须有「①都没有」和「⑪每段都一样」两个出口：
  · 前者用来判断 EL 到底正不正常（不能逼人在干净音频上硬选一段）
  · 后者 = 全局性问题，不是局部，直接换排查方向

## 3 题
| 题 | 用途 |
|---|---|
| d1 我们 9 段 | 定位电子音最重的 10 秒 |
| d2 EL 9 段 | 对照：EL 上是否也标得出 → 确认是不是我们特有的问题 |
| ctrl | 答题有效性校验 |

## 切片口径
统一从成品 **0 秒**起取 90 秒，切成 9 × 10 秒。
所以「第 k 段」= 成品第 (k-1)*10 ~ k*10 秒，不需要任何折算。
"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
TMPD = "/mnt/tts/diag/_z34"
TGT_LUFS = -26.1
SR_CLIP = 24000

OURS = "/mnt/tts/output/longprobe-z32-6cdd/longprobe-z32-6cdd-1.mp3"
EL_S = "/mnt/tts/data/cases/6cdd25bc75f2/merged.wav"

START = 0.0        # 🔴 从成品 0 秒起 → 段号即真实秒数区间
TOTAL = 90.0
N_SEG = 9
SEG = TOTAL / N_SEG
SUFFIX = "20261016z3"

CIRC = "①②③④⑤⑥⑦⑧⑨⑩⑪"


def lufs(p):
    r = subprocess.run(["ffmpeg", "-i", p, "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    v = [l for l in r.stderr.splitlines() if "I:" in l and "LUFS" in l]
    return float(v[-1].split("I:")[1].split("LUFS")[0].strip()) if v else None


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()[:8]


def hhmmss(t):
    return "%02d:%02d" % (int(t) // 60, int(t) % 60)


def seg_slice(src, k):
    """取第 k 段（k 从 1 起）"""
    s0 = START + (k - 1) * SEG
    t = os.path.join(TMPD, "_s%d.wav" % k)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-ss", "%.1f" % s0, "-ar", str(SR_CLIP), "-ac", "1",
                    "-t", "%.1f" % SEG, t], check=True)
    got = os.path.getsize(t) / 2.0 / SR_CLIP
    if got < SEG - 0.5:
        sys.exit("切片越界：%s 第%.0fs 起取%.0fs，实际仅%.1fs"
                 % (os.path.basename(src), s0, SEG, got))
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


def seg_item(iid, who, src, prefix, extra_desc, note):
    """一段一播放器的单选题。选项 A=都没有，B..=第 1..N 段，最后=每段都一样。"""
    opts = [{"key": "A", "file": None,
             "label": "%s 都没有，%d 段听起来都正常" % (CIRC[0], N_SEG)}]
    for k in range(1, N_SEG + 1):
        opts.append({
            "key": chr(ord("B") + k - 1), "file": "%s_s%02d.mp3" % (prefix, k),
            "label": "%s 第 %d 段（%s–%s）" % (CIRC[k], k,
                                            hhmmss(START + (k - 1) * SEG),
                                            hhmmss(START + k * SEG))})
    opts.append({"key": chr(ord("B") + N_SEG), "file": None,
                 "label": "%s 每一段都差不多，分不出哪段更严重" % CIRC[N_SEG]})
    return {
        "id": iid, "dim": "电子音定位（%s，%d 段）" % (who, N_SEG),
        "title": "【%s】这 %d 段，哪一段的「电子感」最明显？" % (who, N_SEG),
        "desc": ("下面 %d 段音频，每段 %d 秒，**按顺序排好了**"
                 "（第 1 段 = 成品第 %s 秒到 %s 秒）。\n\n"
                 "**按顺序各听一遍**，然后只点一下你觉得**电子感最明显**"
                 "的那一段。\n"
                 "不需要暂停、不需要计时、不需要打字。\n\n"
                 "· 如果听下来都觉得正常 → 选「%s 都没有」\n"
                 "· 如果每段都一样糟、分不出高下 → 选「%s 每段都一样」\n\n"
                 "%s"
                 % (N_SEG, int(SEG), hhmmss(START), hhmmss(TOTAL),
                    CIRC[0], CIRC[N_SEG], extra_desc)),
        "ref": None,
        "note": note,
        "options": opts}


def main():
    for p in (OURS, EL_S):
        if not os.path.exists(p):
            print("缺素材：%s" % p)
            return 1
    os.makedirs(TMPD, exist_ok=True)
    made = {}

    for tag, src, prefix in (("d1", OURS, "z34_our"), ("d2", EL_S, "z34_el")):
        for k in range(1, N_SEG + 1):
            made["%s_s%02d" % (prefix, k)] = emit(
                seg_slice(src, k), "%s_s%02d" % (prefix, k),
                "%s seg%d" % (tag, k))

    ctrl = seg_slice(OURS, 3)
    made["z34_ctrl1"] = emit(ctrl, "z34_ctrl1", "ctrl A")
    made["z34_ctrl2"] = emit(ctrl, "z34_ctrl2", "ctrl B")

    print("\n=== md5 自检 ===")
    inv = {}
    for nm, p in made.items():
        inv.setdefault(md5(p), []).append(nm)
    bad = False
    for h, ns in inv.items():
        if len(ns) > 1:
            ok = set(ns) <= {"z34_ctrl1", "z34_ctrl2"}
            print("  %-9s %s" % (h, "OK(控制题同素材)" if ok else "撞车 -> %s" % ns))
            if not ok:
                bad = True
    if not bad:
        print("  无撞车")

    items = [
        seg_item("d1_seg_ours_%s" % SUFFIX, "我们的当前线上版本", OURS, "z34_our",
                 "这是**我们的**声音（男声）。\n"
                 "重点听「**电子感**」——发假、像机器合成、像老收音机那种感觉。",
                 "动作：拿到段号 k → 直接切出成品第 (k-1)*10 ~ k*10 秒那段波形，\n"
                 "      逐帧看频谱/包络/谐波结构，定位电子音的物理来源。\n"
                 "      🔴 一次定位精度只有 10 秒，不满意下轮在那 10 秒里再切 10 段。\n"
                 "      选「每段都一样」→ 不是局部问题，是全局/引擎层面，改排查方向。"),
        seg_item("d2_seg_el_%s" % SUFFIX, "EL（对照）", EL_S, "z34_el",
                 "这是 **EL** 的声音（同一段文案）。\n"
                 "**也请按同样标准听一遍**——这一步很关键。",
                 "🔴 这一题是对照，用来判断 EL 到底正不正常：\n"
                 "  选「都没有」→ EL 确实干净 → 我们标出的问题是我们特有的\n"
                 "  选某一段    → EL 也有问题 → 说明员工的判断标准/听音环境有问题，\n"
                 "                或者这段文案本身难念，得先校准听感\n"
                 "  选「每段都一样」→ 两段都全局性电子，那要怀疑对比方式"),
        {"id": "ctrl_same_%s" % SUFFIX, "dim": "答题有效性校验",
         "title": "【最后一题】这两段，听起来有区别吗？",
         "desc": "**这道题必须选一个**（选错也没关系）。\n\n"
                 "这两段是**同一段音频放了两遍**，内容完全一样。听 5 秒就够。",
         "ref": None,
         "note": "隐藏控制题：正解=C 两段一样。\n"
                 "沿用问法「这两段听起来有区别吗」——已连续 4 轮全对；\n"
                 "旧问法「哪段念得更快」4 人里 2 人错，别再用。",
         "options": [
             {"key": "A", "file": "z34_ctrl1.mp3", "label": "A 明显有区别"},
             {"key": "B", "file": "z34_ctrl2.mp3", "label": "B 明显有区别"},
             {"key": "C", "file": None, "label": "C 没有区别，是一样的"}]},
    ]

    print("\n=== 题目自检 ===")
    ok = True
    for it in items:
        n = sum(1 for o in it["options"] if o.get("file"))
        r = (it.get("ref") or {}).get("file")
        segs = [o for o in it["options"]
                if o.get("file") and "_s" in (o.get("file") or "")]
        if it["id"].startswith("d1_") or it["id"].startswith("d2_"):
            if len(segs) != N_SEG:
                print("  %s 段数 %d != %d" % (it["id"], len(segs), N_SEG))
                ok = False
            nof = [o for o in it["options"] if not o.get("file")]
            if len(nof) < 2:
                print("  %s 缺出口选项（都没有 / 每段都一样）" % it["id"])
                ok = False
            # 段号必须严格递增且文件存在
            prev = 0
            for o in it["options"]:
                f = o.get("file") or ""
                if f and "_s" in f:
                    k = int(f.rsplit("_s", 1)[1].split(".")[0])
                    if k != prev + 1:
                        print("  %s 段号不连续：%d 在 %d 后" % (it["id"], k, prev))
                        ok = False
                    prev = k
                    if not os.path.exists(os.path.join(FB_DIR, f)):
                        print("  %s 缺文件 %s" % (it["id"], f))
                        ok = False
        if n == 1:
            print("  %s 单音频题必须改用 ref" % it["id"])
            ok = False
        if n == 0 and not r:
            print("  %s 既无选项音频也无 ref -> 废题" % it["id"])
            ok = False
    if ok:
        print("  题目结构 OK（%d 段/题，出口选项齐全，段号连续）" % N_SEG)

    out = os.path.join(FB_DIR, "items.json")
    arc = os.path.join(FB_DIR, "items_roundz2.json")
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
