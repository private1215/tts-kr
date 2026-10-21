#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 z37 轮 **盲听裁决**：链路 16k → 24k（v3.7）到底有没有把「发闷」修掉？

## 为什么出这一轮
v3.6 → v3.7 修的是**带宽**：`app/fix_chain.py` 全链路走 `load16()`（16 kHz），
omni 原生 24 kHz 输出一进链就被砍掉 8 kHz 以上 → 成品有效带宽只有 8 kHz。
v3.7 把主链提到 24 kHz，客观已确认（docs/r53）：

| | 8-10k | 10-12k | 音色 vs EL |
|---|---|---|---|
| L3z36（v3.6, 16k） | 2.37e-5 | 1.04e-8（地板） | 0.9356 |
| **L3z37（v3.7, 24k）** | **1.38e-3** | **2.11e-4** | **0.9518** |
| EL 男声标杆 | 8.24e-4 | 1.10e-4 | 1.0000 |

🔴 但 z44/z20 的教训是：**客观改善 ≠ 听得出来**，而且这次 campplus 恰恰
「看不见」8-12k 的损失（它自己在 16k 口径测）。所以必须盲听拍板。

## 🔴 为什么改用「单段独立评分」而不是 A/B 对比
z20 用户纠正过：员工是普通懂韩语的人，让他们做「找不同」的 A/B 对比
既难又不准。z36 又踩了一次坑（选项只有「都没有」没有「都有」，导致
员工把「两段都有问题」挤进 C，被误读成「听不出区别」）。
→ 本轮回到 z20 的做法：**不问差异，问印象**。

## 题面
每段只问一句：「这段听起来像不像真人在说话？」
① 像真人，很自然（2 分） ② 听得出是 AI 但还算舒服（1 分） ③ 明显是机器（0 分）

六段 = 男/女 × {v3.7, v3.6, 标杆}，按性别分组算平均分，三方直接对比。
另加一题 A/B 控制题防瞎选（🔴 选项必须同时有「两段一样」和「两段都不像」）。

用法：TTS_VER=v3.7 TTS_ROUND=z37 venv/bin/python mk_z37.py
"""
import glob
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")
from asr_check import normalize, transcribe  # noqa: E402

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
TMPD = "/mnt/tts/diag/_z37"
TGT_LUFS = -20.0          # z20 口径：单段评分最怕员工凭音量给分
SR = 24000
DUR = 25.0
ASR_SR = 16000

EL = {"male": "/mnt/tts/data/cases/1a6c20a60481/merged.wav",
      "female": "/mnt/tts/data/cases/00b6a45ad6aa/merged.wav"}

VER = os.environ.get("TTS_VER", "v3.7")
ROUND = os.environ.get("TTS_ROUND", "z37")
SIM_MIN = 0.55
SEARCH = (-8, -4, 0, 4, 8)
ASR_MIN_CHARS = 20

# 取样点：避开开头（headfix 区）与结尾，取中段
SPOTS = (150.0, 260.0, 350.0)


def latest(pat):
    """产物目录会被 _unique_dir 加 -N 后缀，按 mtime 取最新。"""
    c = sorted(glob.glob(pat), key=os.path.getmtime)
    return c[-1] if c else None


NEW = {"male": latest("/mnt/tts/output/L3z37-male*/L3z37-male-1.mp3"),
       "female": latest("/mnt/tts/output/L3z37-female*/L3z37-female-1.mp3")}
OLD = {"male": latest("/mnt/tts/output/L3z36-male*/L3z36-male-1.mp3"),
       "female": latest("/mnt/tts/output/L3z36-female*/L3z36-female-1.mp3")}


def qid(slug):
    return "%s-%s-%s" % (VER, ROUND, slug)


def lufs(p):
    r = subprocess.run(["ffmpeg", "-i", p, "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    v = [l for l in r.stderr.splitlines() if "I:" in l and "LUFS" in l]
    return float(v[-1].split("I:")[1].split("LUFS")[0].strip()) if v else None


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()[:8]


def dur_of(p):
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", p], capture_output=True,
        text=True).stdout.strip())


def clip(src, s0, dur=DUR, tag=""):
    os.makedirs(TMPD, exist_ok=True)
    t = os.path.join(TMPD, "_c%s.wav" % hashlib.md5(
        ("%s|%.2f|%.1f" % (src, s0, dur)).encode()).hexdigest()[:8])
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-ss", "%.2f" % max(s0, 0.0), "-ar", str(SR), "-ac", "1",
                    "-t", "%.1f" % dur, t], check=True)
    got = os.path.getsize(t) / 2.0 / SR
    if got < dur - 0.5:
        sys.exit("X 切片越界：%s 第 %.1fs 起只有 %.1fs" % (tag or src, s0, got))
    return t


def hyp(wav):
    return normalize(transcribe(wav, language="Korean"))


def sim(a, b):
    import difflib
    return difflib.SequenceMatcher(None, a[:120], b[:120], autojunk=False).ratio()


def emit(wav, name):
    g = TGT_LUFS - lufs(wav)
    out = os.path.join(FB_DIR, "%s.wav" % name)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav,
                    "-af", "volume=%gdB" % g, "-c:a", "pcm_s16le", out],
                   check=True)
    back = lufs(out)
    print("     %-14s md5=%s LUFS %6.1f %s"
          % (name, md5(out), back or 0,
             "OK" if back and abs(back - TGT_LUFS) < 1.0 else "!!偏差大"))
    return out


def single_q(iid, wav, dim, title, note):
    """单段独立评分：只有 A 带 file，B/C 是评价档位（feedback.html 的坑）。"""
    return {
        "id": iid, "dim": dim, "title": title,
        "desc": "听完这一段，凭**第一感觉**选一个。\n不用反复听，也不用跟别的段比较。",
        "ref": None, "note": note, "ver": VER,
        "options": [
            {"key": "A", "file": wav, "label": "① 像真人，很自然"},
            {"key": "B", "file": None, "label": "② 听得出是 AI，但还算舒服"},
            {"key": "C", "file": None, "label": "③ 明显是机器"}]}


def ab_q(iid, a, b, dim, title, desc, note):
    """A/B 控制题。🔴 选项必须同时给「两段一样」和「两段都不像」两个出口。"""
    return {
        "id": iid, "dim": dim, "title": title, "desc": desc,
        "ref": None, "note": note, "ver": VER,
        "options": [
            {"key": "A", "file": a, "label": "① 第 1 段更像真人"},
            {"key": "B", "file": b, "label": "② 第 2 段更像真人"},
            {"key": "C", "file": None, "label": "③ 两段听起来一样"},
            {"key": "D", "file": None, "label": "④ 两段都听不出像真人"}]}


def pick_spot(src_a, src_e):
    """在中段挑一个 ASR 能对上 EL 的位置，保证三方内容相近。"""
    da, de = dur_of(src_a), dur_of(src_e)
    ke = de / da
    for t in SPOTS:
        if t + DUR > da - 1.0:
            continue
        ha = hyp(clip(src_a, t, DUR, tag="A"))
        if len(ha) < ASR_MIN_CHARS:
            continue
        best = None
        for off in SEARCH:
            pos = max(t * ke + off, 0.0)
            s = sim(ha, hyp(clip(src_e, pos, DUR, tag="EL")))
            if best is None or s > best[1]:
                best = (pos, s)
        print("   位置 %.1fs → 标杆 %.1fs 相似度 %.3f" % (t, best[0], best[1]))
        if best[1] >= SIM_MIN:
            return t, best[0], best[1]
    return None


def main():
    os.makedirs(TMPD, exist_ok=True)
    for g in ("male", "female"):
        for k, d in (("NEW", NEW), ("OLD", OLD)):
            p = d[g]
            if not p or not os.path.exists(p):
                print("X 缺素材 %s/%s：%s" % (g, k, p))
                return 1
            print("%s %s = %s" % (g, k, os.path.basename(p)))

    made, info = {}, {}
    for g in ("male", "female"):
        print("\n=== %s：挑取样点（ASR 对齐到标杆）===" % g)
        got = pick_spot(NEW[g], EL[g])
        if got is None:
            print("X %s 对不齐，不出题" % g)
            return 1
        t_new, t_el, s = got
        # v3.6 与 v3.7 同文案同 seed，时长几乎一致，按时间比折算
        k_old = dur_of(OLD[g]) / dur_of(NEW[g])
        t_old = t_new * k_old
        info[g] = (t_new, t_old, t_el, s)
        print("   ✅ %s  v3.7 %.1fs / v3.6 %.1fs / 标杆 %.1fs  相似度 %.3f"
              % (g, t_new, t_old, t_el, s))
        for slug, src, t in (("new", NEW[g], t_new), ("old", OLD[g], t_old),
                             ("el", EL[g], t_el)):
            nm = "z37_%s_%s_%s" % (g[0], slug, ROUND)
            made[nm] = emit(clip(src, t, DUR, tag=slug), nm)

    # ---- 控制题：同一段放两遍（A/B）----
    print("\n=== 控制题（同一段放两遍）===")
    made["z37_ctrl1"] = emit(clip(NEW["male"], 60.0, DUR, tag="CTRL"),
                             "z37_ctrl1")
    made["z37_ctrl2"] = emit(clip(NEW["male"], 60.0, DUR, tag="CTRL"),
                             "z37_ctrl2")

    print("\n=== md5 自检（防素材撞车）===")
    inv = {}
    for nm, p in made.items():
        inv.setdefault(md5(p), []).append(nm)
    bad = False
    for h, ns in inv.items():
        if len(ns) > 1 and "ctrl" not in ns[0]:
            print("  X 撞车 %s -> %s" % (h, ns))
            bad = True
    if not bad:
        print("  OK 无撞车（ctrl1/ctrl2 刻意相同）")

    def nm(g, slug):
        return "z37_%s_%s_%s.wav" % (g[0], slug, ROUND)

    items = [
        single_q(qid("m_new"), nm("male", "new"), "男声·整体印象",
                 "这段听起来，像不像真人在说话？",
                 "取样 %.1fs 起 25s。\n"
                 "拿到答案我改什么：和下面「男声旧版」「男声标杆」的分一起看 ——\n"
                 "  新版分 ≥ 标杆分 → 带宽问题已解决，该维度收工。\n"
                 "  新版分 > 旧版分但仍 < 标杆分 → 方向对，继续找剩余差距。\n"
                 "  新版分 ≈ 旧版分 → 24k 的改善听不出来，按 z44 封存，别再加码。"
                 % info["male"][0]),

        single_q(qid("m_old"), nm("male", "old"), "男声·整体印象（换一段）",
                 "这段听起来，像不像真人在说话？",
                 "取样 %.1fs 起 25s（与上一题同一段话，可作对照）。\n"
                 "这一段的分数用来判断「新版到底有没有变好」。"
                 % info["male"][1]),

        single_q(qid("m_el"), nm("male", "el"), "男声·整体印象（再换一段）",
                 "这段听起来，像不像真人在说话？",
                 "取样 %.1fs 起 25s。这是我们想追上的水平（%d%% 相似度对齐）。"
                 % (info["male"][2], round(info["male"][3] * 100))),

        single_q(qid("f_new"), nm("female", "new"), "女声·整体印象",
                 "这段听起来，像不像真人在说话？",
                 "取样 %.1fs 起 25s。\n"
                 "拿到答案我改什么：与「女声旧版」「女声标杆」的分一起看，判据同男声。"
                 % info["female"][0]),

        single_q(qid("f_old"), nm("female", "old"), "女声·整体印象（换一段）",
                 "这段听起来，像不像真人在说话？",
                 "取样 %.1fs 起 25s（与上一题同一段话，可作对照）。"
                 % info["female"][1]),

        single_q(qid("f_el"), nm("female", "el"), "女声·整体印象（再换一段）",
                 "这段听起来，像不像真人在说话？",
                 "取样 %.1fs 起 25s。这是我们想追上的水平（%d%% 相似度对齐）。"
                 % (info["female"][2], round(info["female"][3] * 100))),

        ab_q(qid("ctrl"), "z37_ctrl1.wav", "z37_ctrl2.wav", "控制题",
             "这两段，**哪一段更像真人在说话**？",
             "听一遍就选，凭第一感觉。",
             "🔴 **控制题：①和②是同一段音频**（男声 60s 处 25s）。\n"
             "若多数人选①或② → 本轮存在瞎选 / 位置偏好，**整份作废**需重出。\n"
             "选③ = 判别力正常，本轮结论可用。\n"
             "选④ = 两段都被判「不像真人」，说明素材本身有问题，也需重出。"),
    ]

    print("\n=== 盲听自检 ===")
    banned = ("我们的", "我们", "EL", "标杆", "参考", "标准", "v3.7", "v3.6",
              "omnivoice", "16k", "24k", "带宽", "高频", "采样", "click",
              "load16", "fix_chain")
    leak = []
    for it in items:
        if "-ctrl" in it["id"]:
            continue
        for f in ("title", "desc", "dim"):
            for w in banned:
                if w in it[f]:
                    leak.append((it["id"], f, w))
        for o in it["options"]:
            for w in banned:
                if w in o["label"]:
                    leak.append((it["id"], o["key"], w))
    if leak:
        for x in leak:
            print("  X 泄题 %s %s 含 %r" % x)
        return 1
    print("  OK 题面与选项无身份词（dim 也查了，它同样显示在卡片上）")

    out = os.environ.get("TTS_ITEMS_OUT", os.path.join(FB_DIR, "items_z37.json"))
    json.dump({"items": items}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n已写入 %s（%d 题，ver=%s / round=%s）" % (out, len(items), VER, ROUND))
    chk = json.load(open(out, encoding="utf-8"))
    assert isinstance(chk, dict) and "items" in chk
    print("  OK 格式自检：dict.items 有 %d 题" % len(chk["items"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
