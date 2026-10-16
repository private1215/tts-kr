#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第47轮（z15）**男声修复复验**：修完之后，员工还听得出问题吗？

## 为什么要出这一轮
员工在 18:53 那条（gs47m6_e10b，男声 GS）报了三条：
    「杂音 + 文本文字说话不匹配 乱说话了 + 说话速度慢/一会快一会慢」

## 已客观修复（L3 长篇实测，1500 字）
| 指标 | 修复前 | 修复后 |
|---|---|---|
| cps | 2.03 | **6.92**（EL 6.71） |
| CER | 0.6606 | **0.0869** |
| 乱说话(大段 insert) | 6 次 | **0 次** |

根因：**GS 参考音频与 ref_text 不匹配**（10s 音频配 160 字文本，
隐含语速 16 字/s，模型被迫拖长每个字，并把参考文本当正文反复念）。
已改成句边界切片 7.55s / 51 字（6.75 字/s）。

## 但客观达标 ≠ 员工满意（第44轮教训）
物理上量到的差异 ≠ 员工能听到的差异。所以必须回来问一次：
  · 语速 6.92 vs EL 6.71（我们**快 3%**）——会不会显得急？**这是本轮最大不确定点**
  · 杂音还在不在？（上轮报的，本轮未针对处理，需要确认是否随语速修复一并消失）

## 设计
素材 = **wav**（fb 只按文件名取，不编码 mp3，排除编码变量）。
按文本对齐（ASR 实测：EL 时间 ≈ 我们时间 × 0.85）取两个切点：
  s1 = 开头（我们 0s / EL 0s），s2 = 中段（我们 120s / EL 102s）。
A 恒为**我们的**，B 恒为 **EL**，但**选项里不标身份**（盲听铁律）。

用法：/mnt/tts/projects/qwen3-tts/venv/bin/python mk_z47.py
"""
import hashlib
import json
import os
import subprocess
import sys

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
TMPD = "/mnt/tts/diag/_z47"
TGT_LUFS = -26.1
SR = 24000
DUR = 20.0

EL = "/mnt/tts/data/cases/6cdd25bc75f2/merged.wav"
CUR = "/mnt/tts/output/gs47m8_final/gs47m8_final-1.mp3"
CURW = os.path.join(TMPD, "cur.wav")

# ASR 实测对齐：EL 时间 ≈ 我们时间 × 0.85（0->0 / 60->51 / 120->102 / 180->153）
CLIPS = [
    ("z47_a_s1", CURW, 0.0),
    ("z47_b_s1", EL, 0.0),
    ("z47_a_s2", CURW, 120.0),
    ("z47_b_s2", EL, 102.0),
]
SUFFIX = "20261016z15"


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
                    "-ar", str(SR), "-ac", "1", "-t", "%.1f" % dur, t], check=True)
    got = os.path.getsize(t) / 2.0 / SR
    if got < dur - 0.5:
        sys.exit("切片越界：%s 第%.1fs 起，实际%.1fs" % (src, s0, got))
    return t


def emit(wav, name):
    """输出 wav，不编码 mp3（排除编码变量）。"""
    g = TGT_LUFS - lufs(wav)
    out = os.path.join(FB_DIR, "%s.wav" % name)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav,
                    "-af", "volume=%gdB" % g,
                    "-c:a", "pcm_s16le", out], check=True)
    back = lufs(out)
    print("  %-10s md5=%s  LUFS %6.1f %s"
          % (name, md5(out), back if back else 0,
             "OK" if back and abs(back - TGT_LUFS) < 1.0 else "!!爆表"))
    return out


def two_q(iid, a, b, title, desc, note, la, lb):
    """两段比一项：① 第1段 / ② 第2段 / ③ 差不多。不标身份。"""
    return {
        "id": iid, "dim": "男声·修复复验",
        "title": title,
        "desc": desc,
        "ref": None, "note": note,
        "options": [
            {"key": "A", "file": a, "label": la},
            {"key": "B", "file": b, "label": lb},
            {"key": "C", "file": None, "label": "③ 差不多 / 听不出明显区别"}]}


def main():
    os.makedirs(TMPD, exist_ok=True)
    if not os.path.exists(CURW):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", CUR,
                        "-ar", str(SR), "-ac", "1", CURW], check=True)
    if not os.path.exists(EL):
        print("缺素材：%s" % EL)
        return 1

    made = {}
    print("=== 切片 + 归一（20s，输出 wav）===")
    for nm, src, t in CLIPS:
        made[nm] = emit(clip(src, t, DUR), nm)
    made["z47_ctrl1"] = emit(clip(CURW, 60.0, 6.0), "z47_ctrl1")
    made["z47_ctrl2"] = emit(clip(CURW, 60.0, 6.0), "z47_ctrl2")

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

    items = [
        two_q("z15_sp1_%s" % SUFFIX, "z47_a_s1.wav", "z47_b_s1.wav",
              "这两段，**说话的速度**哪一段更自然？",
              "两段读的是同一段话，声音不一样是正常的。\n"
              "**只问一件事：哪一段说话的快慢听起来更舒服、更自然？**\n"
              "不用管声音像不像，只听节奏和快慢。\n"
              "听一遍就选，凭第一感觉。",
              "语速复验①（开头）：①=我们（修后 GS 男声，cps 6.92）②=EL（cps 6.71）。\n"
              "我们比 EL **快 3%**。上一轮员工报「说话慢、一会快一会慢」，\n"
              "修复前 cps 只有 2.03（慢 3 倍），现在 6.92。\n"
              "选② → 我们还是偏急 → 把参考音频语速从 6.75 字/s 往下调（改用 P1 的 7.34→更慢切片）。\n"
              "选①或③ → 语速问题解决，本项封存。",
              "① 第 1 段更自然", "② 第 2 段更自然"),

        two_q("z15_sp2_%s" % SUFFIX, "z47_a_s2.wav", "z47_b_s2.wav",
              "换一段话再听一次：**说话的速度**哪一段更自然？",
              "还是同样的问法，换了一段内容。\n"
              "**只问哪一段说话的快慢更舒服。**\n"
              "如果和上一题感觉一样，按第一感觉选就行。",
              "语速复验②（中段，我们 120s / EL 102s，ASR 已核对是同一句）。\n"
              "换位置复验，排除「只有开头碰巧正常」。",
              "① 第 1 段更自然", "② 第 2 段更自然"),

        {"id": "z15_noise_%s" % SUFFIX, "dim": "男声·杂音复验",
         "title": "这两段里，**有没有哪一段能听到杂音**？",
         "desc": ("两段读的是同一段话。\n"
                  "**只问一件事：有没有杂音？**\n"
                  "比如沙沙声、嘶嘶声、滋滋声、嗡嗡声，或者音量一阵一阵的波纹感。\n"
                  "听一遍就选，凭第一感觉。"),
         "ref": None,
         "note": ("杂音复验：①=我们（修后）②=EL。上一轮员工同时报了「杂音 + 乱说话」。\n"
                  "乱说话已客观解决（CER 0.66->0.087，大段 insert 6->0），\n"
                  "但**杂音没有单独处理过**，只是可能随着语速修复一并消失。\n"
                  "选① → 杂音还在 → 下一步做 silfloor A/B（TTS_SILFLOOR=0）。\n"
                  "选② → EL 反而被指认有杂音 → 说明判别力或素材有问题，本轮结论作废。\n"
                  "选③ → 杂音随语速修复消失，本项封存。"),
         "options": [
             {"key": "A", "file": "z47_a_s1.wav", "label": "① 第 1 段有杂音"},
             {"key": "B", "file": "z47_b_s1.wav", "label": "② 第 2 段有杂音"},
             {"key": "C", "file": None, "label": "③ 两段都没有杂音"},
             {"key": "D", "file": None, "label": "④ 两段都有杂音"},
             {"key": "E", "file": None, "label": "⑤ 说不准 / 没听出来"}]},

        {"id": "z15_txt_%s" % SUFFIX, "dim": "男声·内容复验",
         "title": "这两段里，有没有**念错、重复、或者念出文案里没有的话**？",
         "desc": ("两段读的是同一段话。\n"
                  "**只问内容对不对**：有没有多念、少念、重复念某一句，\n"
                  "或者冒出文案里根本没有的句子。\n"
                  "听一遍就选。"),
         "ref": None,
         "note": ("内容复验：①=我们 ②=EL。上一轮员工报「文本文字说话不匹配、乱说话了」。\n"
                  "客观已解决（CER 0.6606->0.0869；反复念参考文本的大段 insert 6 次 -> 0 次）。\n"
                  "选① → 还有内容问题 → 继续查参考音频/ref_text 对齐。\n"
                  "选③ → 内容问题解决，本项封存。"),
         "options": [
             {"key": "A", "file": "z47_a_s2.wav", "label": "① 第 1 段有"},
             {"key": "B", "file": "z47_b_s2.wav", "label": "② 第 2 段有"},
             {"key": "C", "file": None, "label": "③ 两段都没有"},
             {"key": "D", "file": None, "label": "④ 两段都有"},
             {"key": "E", "file": None, "label": "⑤ 说不准 / 没听出来"}]},

        {"id": "z15_ctrl_%s" % SUFFIX, "dim": "答题有效性校验",
         "title": "【最后一题】这两段，听起来有区别吗？",
         "desc": "**这道题必须选一个**（选错也没关系）。\n\n"
                 "这两段是**同一段音频放了两遍**，内容完全一样。听 5 秒就够。",
         "ref": None, "note": "隐藏控制题：正解 = C 两段一样。答错整份作废。",
         "options": [
             {"key": "A", "file": "z47_ctrl1.wav", "label": "A 明显有区别"},
             {"key": "B", "file": "z47_ctrl2.wav", "label": "B 明显有区别"},
             {"key": "C", "file": None, "label": "C 没有区别，是一样的"}]},
    ]

    print("\n=== 题目自检 ===")
    ok = True
    banned = ("我们的", "我们", "EL", "标杆", "参考", "标准")
    for it in items:
        n = sum(1 for o in it["options"] if o.get("file"))
        for o in it["options"]:
            if o.get("file") and not os.path.exists(os.path.join(FB_DIR, o["file"])):
                print("  %s 缺文件 %s" % (it["id"], o["file"]))
                ok = False
        if n == 0:
            print("  %s 无音频选项" % it["id"])
            ok = False
        # 🔴 盲听铁律：选项标签 / 标题 / 描述里不许出现身份词
        for f in ("title", "desc"):
            for w in banned:
                if w in (it.get(f) or ""):
                    print("  %s 的 %s 泄露身份词 %r" % (it["id"], f, w))
                    ok = False
        for o in it["options"]:
            for w in banned:
                if w in (o.get("label") or ""):
                    print("  %s 选项 %s 泄露身份词 %r" % (it["id"], o["key"], w))
                    ok = False
    print("  自检通过" if ok else "  !! 自检未通过，不要发布")

    out = os.path.join(FB_DIR, "items.json")
    # 🔴 服务端按 dict 读（items_roundz13.json 就是 {"items": [...]}）。
    #    写成裸 list 会让 /api/fb/where 抛 'list' object has no attribute 'get'，
    #    整个反馈台打不开 —— 自检里必须查这一步。
    json.dump({"items": items}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n已写出 %s（%d 题）" % (out, len(items)))
    for it in items:
        print("  -", it["id"], "|", it["title"][:40])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
