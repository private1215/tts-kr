#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第47轮 **z17 男声路线裁决**：GS（质量好但慢 9.5%）还是 voxcpm（语速准但噪）？

## 客观数据（同文案 6cdd25bc75f2，16k 统一口径）
| 指标 | EL 男声标杆 | voxcpm 现役 | **GS（yohan_gs_f8）** |
|---|---|---|---|
| 静音段 4-8k 杂音 | 10.968 | 18.201 | **11.077** |
| click % | 0.259 | 1.175 | **0.388** |
| 音色余弦（同长度） | 1.000 | 0.8730 | **0.9576**（同人上限 0.9628）|
| CER | — | — | **0.0624**（零大段 insert）|
| **cps** | **8.23** | **8.36** ✅ | 7.45 ❌ 慢 9.5% |

## 🔴 但「加速」这条路走不通，已实测否决
把 cps 从 7.45 提到 8.39 需要 atempo 加速 11%。实测加速后：
**自适应 EQ 判定高频不足 → highshelf 顶格 +6dB → 静音段杂音 11.077 → 35.986**
（EL 10.968 的 3.3 倍）。WSOLA 时域压缩平滑掉高频，EQ 又把它补回来，
补的方式是抬静音段底噪 → 嘶声爆炸。**所以男声 GS 只能维持 7.45 cps。**

## 本轮要问的（都是「拿到答案就能定方案」的题）
1. **语速**：GS 7.45 vs EL 8.23，慢 9.5% 员工听不听得出？
   · 听不出 → 男声直接换 GS（音色/杂音/click 全面碾压）
   · 听得出 → 只能留 voxcpm，或继续找更快的 GS 参考
2. **杂音**：voxcpm 现役的杂音（客观是 EL 的 1.66 倍）员工是否真能听出？
   · 听不出 → 说明杂音这项不是瓶颈，优先换 GS
3. **整体**：GS vs voxcpm，哪个更像真人？

## 素材对齐
同文案 6cdd25bc75f2 前 1500 字。三条 cps 不同，按**字数**换算切点：
  第 800 字 -> GS 107.4s / EL 97.2s / voxcpm 95.7s
（发题前用 tools/alignchk48.py 跑 ASR 确认真的是同一句）

用法：TTS_VER=v1.1 /mnt/tts/projects/qwen3-tts/venv/bin/python mk_z49.py
"""
import hashlib
import json
import os
import subprocess
import sys

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
TMPD = "/mnt/tts/diag/_z49"
TGT_LUFS = -26.1
SR = 24000
DUR = 20.0

GS = "/mnt/tts/output/gs48m_f8/gs48m_f8-1.mp3"      # 7.45 cps
VOX = "/mnt/tts/output/v10_m_vox/v10_m_vox-1.mp3"   # 8.36 cps
EL = "/mnt/tts/data/cases/6cdd25bc75f2/merged.wav"  # 8.23 cps

# 按「第 800 字」换算：t = 800 / cps
CLIPS = [
    ("z49_gs_s1", GS, 0.0),
    ("z49_vox_s1", VOX, 0.0),
    ("z49_el_s1", EL, 0.0),
    ("z49_gs_s2", GS, 800 / 7.45),
    ("z49_vox_s2", VOX, 800 / 8.36),
    ("z49_el_s2", EL, 800 / 8.23),
]

VER = os.environ.get("TTS_VER", "v1.1")
ROUND = os.environ.get("TTS_ROUND", "z17")


def qid(slug):
    return "%s-%s-%s" % (VER, ROUND, slug)


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
    g = TGT_LUFS - lufs(wav)
    out = os.path.join(FB_DIR, "%s.wav" % name)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav,
                    "-af", "volume=%gdB" % g, "-c:a", "pcm_s16le", out],
                   check=True)
    back = lufs(out)
    print("  %-12s md5=%s  LUFS %6.1f %s"
          % (name, md5(out), back if back else 0,
             "OK" if back and abs(back - TGT_LUFS) < 1.0 else "!!爆表"))
    return out


def two_q(iid, a, b, title, desc, note):
    return {
        "id": iid, "dim": "男声·路线裁决",
        "title": title, "desc": desc,
        "ref": None, "note": note, "ver": VER,
        "options": [
            {"key": "A", "file": a, "label": "① 第 1 段"},
            {"key": "B", "file": b, "label": "② 第 2 段"},
            {"key": "C", "file": None, "label": "③ 差不多 / 听不出明显区别"}]}


def main():
    os.makedirs(TMPD, exist_ok=True)
    for p in (GS, VOX, EL):
        if not os.path.exists(p):
            print("缺素材：%s" % p)
            return 1

    made = {}
    print("=== 切片 + 归一（20s，输出 wav）===")
    for nm, src, t in CLIPS:
        made[nm] = emit(clip(src, t, DUR), nm)
    made["z49_ctrl1"] = emit(clip(GS, 60.0, 6.0), "z49_ctrl1")
    made["z49_ctrl2"] = emit(clip(GS, 60.0, 6.0), "z49_ctrl2")

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
        two_q(qid("spd"), "z49_gs_s1.wav", "z49_el_s1.wav",
              "这两段，**说话的速度**哪一段更自然？",
              "两段读的是同一段话，声音不一样是正常的。\n"
              "**只问一件事：哪一段说话的快慢听起来更舒服、更自然？**\n"
              "不用管声音像不像，只听节奏和快慢。\n"
              "听一遍就选，凭第一感觉。",
              "🔴 **本轮最关键的题**：语速裁决①（开头）。\n"
              "①=男声 GS（cps 7.45）②=EL（cps 8.23）。**我们慢 9.5%**。\n"
              "选② → 语速慢被听出了 → 男声不能换 GS（除非找到更快的参考，"
              "但 EL 标杆里最快的句边界片段只有 8.87 字/s，已用到）。\n"
              "选①或③ → 9.5% 听不出 → **男声可以换 GS**"
              "（音色 0.9576 vs 0.8730、杂音 11.077 vs 18.201、click 0.388% vs 1.175%）。"),

        two_q(qid("spd2"), "z49_gs_s2.wav", "z49_el_s2.wav",
              "换一段话再听一次：**说话的速度**哪一段更自然？",
              "还是同样的问法，换了一段内容。\n"
              "**只问哪一段说话的快慢更舒服。**\n"
              "听一遍就选，凭第一感觉。",
              "语速裁决②（中段，按「第 800 字」换算切点：GS 107.4s / EL 97.2s）。\n"
              "换位置复验，排除「只有开头碰巧正常」。"),

        two_q(qid("noise"), "z49_vox_s1.wav", "z49_el_s1.wav",
              "这两段里，**有没有哪一段能听到杂音**？",
              "两段读的是同一段话。\n"
              "**只问一件事：有没有杂音？**\n"
              "比如沙沙声、嘶嘶声、滋滋声、嗡嗡声，或者音量一阵一阵的波纹感。\n"
              "听一遍就选，凭第一感觉。",
              "杂音裁决（开头）：①=voxcpm 现役 ②=EL。\n"
              "客观：静音段 4-8k 18.201 vs 10.968（**高 66%**）；click 1.175% vs 0.259%。\n"
              "选① → 员工真能听出现役的杂音 → 杂音是瓶颈，优先换 GS。\n"
              "选② → EL 反而被指认 → 判别力或素材有问题，本轮结论作废。\n"
              "选③ → 杂音不可闻 → **杂音这项不是瓶颈**，别再往它加码。"),

        two_q(qid("nat"), "z49_gs_s2.wav", "z49_vox_s2.wav",
              "这两段，**哪一段听起来更像真人在说话**？",
              "两段读的是同一段话。\n"
              "**只问整体感觉**：哪一段更像真人在自然地说话，\n"
              "哪一段更像机器在念稿？\n"
              "听一遍就选，凭第一感觉。",
              "整体自然度（中段）：①=男声 GS ②=voxcpm 现役。**两者都不是 EL**。\n"
              "这是**直接决定换不换引擎**的题：\n"
              "选① → 换 GS；选② → 留 voxcpm；选③ → 两者无差，按客观指标换 GS。\n"
              "⚠️ 若与 spd/spd2 的结论矛盾（GS 被判慢但整体更像真人），\n"
              "说明「9.5% 的语速慢」换来的质量提升**值**，仍倾向换 GS。"),

        two_q(qid("ctrl"), "z49_ctrl1.wav", "z49_ctrl2.wav",
              "这两段，**哪一段听起来更像真人在说话**？",
              "听一遍就选，凭第一感觉。",
              "🔴 **控制题：①和②是同一段音频**（男声 GS，60s 处 6s）。\n"
              "若多数人选①或② → 本轮存在瞎选/位置偏好，**所有结论作废**，需要重出。\n"
              "选③ = 判别力正常，本轮结论可用。"),
    ]

    print("\n=== 盲听自检 ===")
    banned = ("我们的", "我们", "EL", "标杆", "参考", "标准", "GS", "voxcpm", "v1.0", "v1.1")
    leak = []
    for it in items:
        if "ctrl" in it["id"]:
            continue
        for f in ("title", "desc"):
            for w in banned:
                if w in it[f]:
                    leak.append((it["id"], f, w))
        for o in it["options"]:
            for w in banned:
                if w in o["label"]:
                    leak.append((it["id"], o["key"], w))
    if leak:
        for x in leak:
            print("  !! 泄题 %s %s 含 %r" % x)
        return 1
    print("  ✅ 选项与题面无身份词")

    out = os.environ.get("TTS_ITEMS_OUT", os.path.join(FB_DIR, "items_z17.json"))
    # 🔴 服务端按 dict 读，写裸 list 会让整个反馈台打不开
    json.dump({"items": items}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n已写入 %s（%d 题）" % (out, len(items)))
    print("  ⚠️ 未覆盖线上 items.json。切换批次：cp %s %s/items.json" % (out, FB_DIR))
    chk = json.load(open(out, encoding="utf-8"))
    assert isinstance(chk, dict) and len(chk["items"]) == len(items)
    print("  ✅ 格式自检通过")
    print("\n🔴 发题后必须跑：")
    print("   python3 tools/alignchk48.py /mnt/tts/fb/z49_gs_s1.wav /mnt/tts/fb/z49_el_s1.wav \\")
    print("       /mnt/tts/fb/z49_gs_s2.wav /mnt/tts/fb/z49_el_s2.wav \\")
    print("       /mnt/tts/fb/z49_vox_s1.wav /mnt/tts/fb/z49_el_s1.wav \\")
    print("       /mnt/tts/fb/z49_gs_s2.wav /mnt/tts/fb/z49_vox_s2.wav")
    return 0


if __name__ == "__main__":
    sys.exit(main())
