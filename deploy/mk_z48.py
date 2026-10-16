#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第47轮 **z16 女声 v1.1 验证**：极简后处理上线后，员工还听得出杂音吗？

## 为什么出这一轮
v1.1 把女声 `omnivoice` 的后处理**极简化**（只做响度对齐到 -22.6 LUFS，
跳过变速 / 变调 / 切窗 / EQ）。依据：裸输出 F0 275.9 == EL 275.9，
根本不需要变调，链路却每窗跑 asetrate+atempo，F0 一点没动，只把静音段
4-8kHz 噪声从 11.64 抬到 13.00 —— 全程纯亏损。

## L3 全篇客观验收（3415 字，16k 统一口径）
| 指标 | EL 标杆 | v1.0 | **v1.1** |
|---|---|---|---|
| 杂音 静音 4-8k % | 11.371 | 14.596 | **12.952** |
| click % | 0.157 | 0.298 | **0.229** |
| 音色余弦 | 1.000 | 0.8270 | **0.9253** |
| cps | 7.45 | 7.49 | 7.53 |

## 🔴 但客观达标 ≠ 员工满意（第44轮 z44 教训，必须回来问一次）
「物理上量到的差异 ≠ 员工能听到的差异」。z44 那次 13dB 的地板改动，
9 票里只有 1 票指向修后 —— 员工**听不出**。所以 v1.1 也必须回问：
  · 杂音 12.952 vs EL 11.371：**还差 1.58**，员工听不听得出？
  · 音色 0.827 → 0.9253（+0.098）：员工会不会觉得更像了？

## 设计
素材 = **wav**（fb 只按文件名取，不编码 mp3，排除编码变量）。
同文案：`female_full.txt` 3415 字 == 案例 `00b6a45ad6aa` 的文本。
时长对齐：我们 453.2s / EL 459.1s → **EL 时间 ≈ 我们时间 × 1.013**。
A 恒为**我们的（v1.1）**，B 恒为 **EL**，但选项里不标身份（盲听铁律）。

用法：TTS_VER=v1.1 /mnt/tts/projects/qwen3-tts/venv/bin/python mk_z48.py
"""
import hashlib
import json
import os
import subprocess
import sys

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
TMPD = "/mnt/tts/diag/_z48"
TGT_LUFS = -26.1
SR = 24000
DUR = 20.0

EL = "/mnt/tts/data/cases/00b6a45ad6aa/merged.wav"
CUR = "/mnt/tts/output/v11_f_official/v11_f_official-1.mp3"
CURW = os.path.join(TMPD, "cur.wav")

# 时长对齐：我们 453.2s / EL 459.1s -> EL = 我们 × 1.013
K = 1.013
CLIPS = [
    ("z48_a_s1", CURW, 0.0),
    ("z48_b_s1", EL, 0.0),
    ("z48_a_s2", CURW, 120.0),
    ("z48_b_s2", EL, 120.0 * K),
    ("z48_a_s3", CURW, 300.0),
    ("z48_b_s3", EL, 300.0 * K),
]

# 🔴 版本标识（见项目规范「版本标识」+ VERSION 文件）
VER = os.environ.get("TTS_VER", "v1.1")
ROUND = os.environ.get("TTS_ROUND", "z16")


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


def two_q(iid, a, b, title, desc, note):
    return {
        "id": iid, "dim": "女声·v1.1 验证",
        "title": title, "desc": desc,
        "ref": None, "note": note, "ver": VER,
        "options": [
            {"key": "A", "file": a, "label": "① 第 1 段"},
            {"key": "B", "file": b, "label": "② 第 2 段"},
            {"key": "C", "file": None, "label": "③ 差不多 / 听不出明显区别"}]}


def main():
    os.makedirs(TMPD, exist_ok=True)
    if not os.path.exists(CURW):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", CUR,
                        "-ar", str(SR), "-ac", "1", CURW], check=True)
    for p in (EL, CUR):
        if not os.path.exists(p):
            print("缺素材：%s" % p)
            return 1

    made = {}
    print("=== 切片 + 归一（20s，输出 wav）===")
    for nm, src, t in CLIPS:
        made[nm] = emit(clip(src, t, DUR), nm)
    # 控制题：同一段放两次（A/B 源文件相同，用于检验判别力）
    made["z48_ctrl1"] = emit(clip(CURW, 60.0, 6.0), "z48_ctrl1")
    made["z48_ctrl2"] = emit(clip(CURW, 60.0, 6.0), "z48_ctrl2")

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
        two_q(qid("noise1"), "z48_a_s1.wav", "z48_b_s1.wav",
              "这两段里，**有没有哪一段能听到杂音**？",
              "两段读的是同一段话。\n"
              "**只问一件事：有没有杂音？**\n"
              "比如沙沙声、嘶嘶声、滋滋声、嗡嗡声，或者音量一阵一阵的波纹感。\n"
              "听一遍就选，凭第一感觉。",
              "杂音复验①（开头）：①=我们 v1.1 ②=EL。\n"
              "客观：杂音 12.952 vs EL 11.371（全篇 16k 口径），**还差 1.58**，\n"
              "v1.0 是 14.596 —— 缺口已收窄 51%，但没追平。\n"
              "选① → 员工还听得出 → 继续查极简链路里剩下的噪声源（裸输出本身）。\n"
              "选② → EL 反而被指认有杂音 → 判别力或素材有问题，本轮结论作废。\n"
              "选③ → 杂音已不可闻，**本项封存**，不再往杂音方向加码。"),

        two_q(qid("noise2"), "z48_a_s3.wav", "z48_b_s3.wav",
              "换一段话再听一次：**有没有哪一段能听到杂音**？",
              "还是同样的问法，换了一段内容（后段）。\n"
              "**只问有没有杂音。**\n"
              "听一遍就选。",
              "杂音复验②（后段，我们 300s / EL 304s）。\n"
              "换位置复验，排除「只有开头碰巧干净」。\n"
              "若①②都选③ → 杂音封存；若都选① → 杂音是真问题，继续修。"),

        two_q(qid("tone"), "z48_a_s2.wav", "z48_b_s2.wav",
              "这两段，**哪一段的声音更像「原来那个人」**？",
              "两段读的是同一段话，声音不一样是正常的。\n"
              "**只问一件事：哪一段的音色更像原来那个人？**\n"
              "不用管快慢和杂音，只听声音本身像不像。\n"
              "听一遍就选，凭第一感觉。",
              "音色复验（中段，我们 120s / EL 121.6s）。\n"
              "客观：音色余弦 v1.0 0.8270 → **v1.1 0.9253**（EL=1.0），+0.098。\n"
              "campplus 认「同一人」的阈值是 0.75，两条都过，但 v1.1 明显更近。\n"
              "选① → 员工觉得 EL 更像 → 音色仍是短板，下一步靠 GS 微调（0.908 那条线）。\n"
              "选③ → 音色已不可分辨，本项封存。"),

        two_q(qid("nat"), "z48_a_s2.wav", "z48_b_s2.wav",
              "这两段，**哪一段听起来更像真人在说话**？",
              "**只问整体感觉**：哪一段更像真人在自然地说话，\n"
              "哪一段更像机器在念稿？\n"
              "听一遍就选，凭第一感觉。",
              "整体自然度（中段，与 tone 同一对素材，问法不同）。\n"
              "这是**终极验收口径**——员工验收标准是「听不出差别」，"
              "而自然度就是他们最直接的表达。\n"
              "若本轮杂音/音色都选③但自然度选② → 说明还有没量到的维度，\n"
              "需要回 S1 重新定位（别再只盯着频谱指标）。"),

        two_q(qid("ctrl"), "z48_ctrl1.wav", "z48_ctrl2.wav",
              "这两段，**哪一段听起来更像真人在说话**？",
              "听一遍就选，凭第一感觉。",
              "🔴 **控制题：①和②是同一段音频**（我们的 v1.1，60s 处 6s）。\n"
              "若多数人选①或② → 说明本轮存在瞎选/位置偏好，\n"
              "**本轮所有结论作废**，需要重出。\n"
              "选③ = 判别力正常，本轮结论可用。"),
    ]

    # 🔴🔴 自检：盲听铁律。选项/标题里不许出现身份词
    print("\n=== 盲听自检 ===")
    banned = ("我们的", "我们", "EL", "标杆", "参考", "标准", "v1.0", "v1.1")
    leak = []
    for it in items:
        if "ctrl" in it["id"]:
            continue          # 控制题的 note 里需要写明身份，那是给内部看的
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

    # 🔴 默认**不**覆盖线上 items.json —— z15 还在收票。
    #    先落到 items_z16.json，明早切换批次时再 cp 成 items.json。
    out = os.environ.get("TTS_ITEMS_OUT", os.path.join(FB_DIR, "items_z16.json"))
    # 🔴 服务端按 dict 读。写成裸 list 会让 /api/fb/where 抛
    #    'list' object has no attribute 'get'，整个反馈台打不开。
    json.dump({"items": items}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n已写入 %s（%d 题）" % (out, len(items)))
    print("  ⚠️ 线上仍是 items.json（z15 收票中）。切换批次：")
    print("     cp %s %s/items.json" % (out, FB_DIR))

    # 🔴 格式自检：服务端按 dict 读，且必须能取到 items
    chk = json.load(open(out, encoding="utf-8"))
    assert isinstance(chk, dict) and "items" in chk, "必须是 {'items': [...]}"
    assert len(chk["items"]) == len(items)
    print("  ✅ 格式自检通过：dict.items 有 %d 题" % len(chk["items"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
