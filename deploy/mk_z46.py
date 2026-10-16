#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第46轮（z14）**杂音定位**：员工说的「波段杂音」到底在哪一段？

## 为什么要出这一轮
z45 上线后员工在 free 里连着 4 题写「波段杂音」，第 1 题点名 2 号（= GS 段）。
但**只凭这句描述不能动手** —— 必须先定位，否则就是自嗨式优化
（第44轮的教训：物理上量到的差异 ≠ 员工能听到的差异）。

## 已排除 / 已查明的（都是实测）
- ❌ 停顿处 silfloor 填的底噪：GS 停顿 p5 抬了 33dB，但 8 段素材的
  停顿区电平与 EL 原声无显著差异（noise47.py）。
- ❌ 8~16Hz 包络整形：现役 1.74~1.86 / GS 1.80~2.15 / **EL 原声 1.95** ——
  EL 比 GS 还高，不是区分因素（mod47.py）。
- 🔴 **两条根本不是同一条流水线**（重大发现）：
  `NEW_ENGINES = ("voxcpm","omnivoice")`，gptsovits **不在里面** →
  GS 走了 legacy 全链（PBA / 包络整形 / 慢尺度轮廓 / 能量动态 /
  **时长对齐 atempo 0.9308、0.9349** / 首尾静音 atempo 0.931 / 呼吸 /
  响度归一 / roomtone / peakfit），现役 omnivoice 走快路径全部跳过。
  GS 语速偏快 7%，被 atempo 0.93 拉慢 → WSOLA 时域拉伸的嫌疑最大。
- 🔴 mp3 编码失真（源 wav ↔ 成品 mp3 的 SNR，已做增益+延迟对齐）：
  现役 **61.6 / 61.8 dB**，GS **44.0 / 44.1 dB** —— **差 17.6 dB**。
  GS 的信号是 mp3 编码器压不动的那一类（典型 = 非自然的高频/周期性成分）。

## 本轮设计：一步排除 mp3 编码变量
**素材直接用 wav**（fb-audio 只按文件名取，不限扩展名），
只做响度归一，**不再编码 mp3**。这样：
  · wav 下杂音消失 → 是我导出时的 mp3 编码问题 → 改导出方式
  · wav 下杂音还在 → 是引擎/链路的锅 → 改链路（绕开 atempo）

三组对照，其中第 3 组是**关键判据**：
  ① 现役 vs GS（s1）  ② 现役 vs GS（s3）  ③ **现役 vs EL**
  → 若连 EL 也被指认有杂音，那就是**素材导出**的问题，与引擎无关。

用法：/mnt/tts/projects/qwen3-tts/venv/bin/python mk_z46.py
"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
TMPD = "/mnt/tts/diag/_z46"
TGT_LUFS = -26.1
SR = 24000
DUR = 20.0

EL = "/mnt/tts/data/cases/00b6a45ad6aa/merged.wav"
CUR = "/mnt/tts/diag/sola47_f_omnivoice.wav"
GS = "/mnt/tts/diag/sola47_f_gptsovits.wav"

SC_CUR = 458.83 / 459.13
SC_GS = 450.38 / 459.13

# (素材名, 源, EL 时刻)
CLIPS = [
    ("z46_a_s1", CUR, 12.0), ("z46_b_s1", GS, 12.0),
    ("z46_a_s3", CUR, 320.0), ("z46_b_s3", GS, 320.0),
    ("z46_a_el", CUR, 30.0), ("z46_b_el", EL, 30.0),
]
SUFFIX = "20261016z14"


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
                    "-ar", str(SR), "-ac", "1", "-t", "%.1f" % dur, t], check=True)
    got = os.path.getsize(t) / 2.0 / SR
    if got < dur - 0.5:
        sys.exit("切片越界：%s 第%.1fs 起，实际%.1fs" % (src, s0, got))
    return t


def emit(wav, name):
    """🔴 输出 **wav**，不编码 mp3 —— 本轮就是要排除编码变量。"""
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


def noise_q(iid, a, b, note):
    return {
        "id": iid, "dim": "女声·杂音定位",
        "title": "这两段的**声音干净程度**，一样吗？",
        "desc": ("两段读的是同一段话。\n"
                 "**只问一件事：有没有哪一段能听到杂音？**\n"
                 "比如沙沙声、嘶嘶声、嗡嗡声，或者音量一阵一阵的波纹感。\n"
                 "听一遍就选，凭第一感觉。"),
        "ref": None, "note": note,
        "options": [
            {"key": "A", "file": a, "label": "① 第 1 段有杂音"},
            {"key": "B", "file": b, "label": "② 第 2 段有杂音"},
            {"key": "C", "file": None, "label": "③ 两段都没有杂音"},
            {"key": "D", "file": None, "label": "④ 两段都有杂音"},
            {"key": "E", "file": None, "label": "⑤ 说不准 / 没听出来"}]}


def main():
    for p in (EL, CUR, GS):
        if not os.path.exists(p):
            print("缺素材：%s" % p)
            return 1
    os.makedirs(TMPD, exist_ok=True)
    made = {}
    print("=== 切片 + 归一（20s，直接输出 wav，不编码 mp3）===")
    for nm, src, t in CLIPS:
        made[nm] = emit(clip(src, scale(src, t), DUR), nm)
    made["z46_ctrl1"] = emit(clip(CUR, 200.0, 6.0), "z46_ctrl1")
    made["z46_ctrl2"] = emit(clip(CUR, 200.0, 6.0), "z46_ctrl2")

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
        noise_q("z14_n1_%s" % SUFFIX, "z46_a_s1.wav", "z46_b_s1.wav",
                "定位①：现役 omnivoice(①) vs GS 微调 SOLA(②)，切点 = 全文第 12s。\n"
                "🔴 素材是 **wav**（不编码 mp3），所以本轮听不出杂音 = 上轮的锅在编码。\n"
                "选② → GS 段确实有杂音 → 改链路（GS 走 legacy 全链，被 atempo 0.93 拉慢，\n"
                "      WSOLA 时域拉伸的嫌疑最大；mp3 编码 SNR 现役 61.6dB vs GS 44.0dB）。\n"
                "选① → 是现役的问题，GS 反而干净。\n"
                "选③ → 上轮的「杂音」是 mp3 编码造成的错觉 → 改导出方式即可，引擎没问题。"),
        noise_q("z14_n2_%s" % SUFFIX, "z46_a_s3.wav", "z46_b_s3.wav",
                "定位②：切点 = 全文第 320s（后段复验）。\n"
                "GS 第 2 块 atempo 0.9349，同为 7% 拉伸。换位置还能不能复现。"),
        noise_q("z14_n3_%s" % SUFFIX, "z46_a_el.wav", "z46_b_el.wav",
                "🔴 **关键判据**：① = 现役（我们），② = **EL 原声**（标杆）。\n"
                "切点 = 全文第 30s。\n"
                "EL 是标杆，理论上最干净。若员工也把 ② 指认为有杂音，\n"
                "→ 说明「杂音」来自**素材导出链路**（与引擎无关），\n"
                "   上轮的结论全部作废，改导出方式重来。\n"
                "若 ③（都没有）→ 说明判别力正常，定位①②的票才可信。"),
        {"id": "z14_ctrl_same_%s" % SUFFIX, "dim": "答题有效性校验",
         "title": "【最后一题】这两段，听起来有区别吗？",
         "desc": "**这道题必须选一个**（选错也没关系）。\n\n"
                 "这两段是**同一段音频放了两遍**，内容完全一样。听 5 秒就够。",
         "ref": None, "note": "隐藏控制题：正解 = C 两段一样。答错整份作废。",
         "options": [
             {"key": "A", "file": "z46_ctrl1.wav", "label": "A 明显有区别"},
             {"key": "B", "file": "z46_ctrl2.wav", "label": "B 明显有区别"},
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
        print("  题目结构 OK（杂音定位×3 + 控制×1），全盲听，wav 素材")

    out = os.path.join(FB_DIR, "items.json")
    arc = os.path.join(FB_DIR, "items_roundz13.json")
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
