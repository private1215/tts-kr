# -*- coding: utf-8 -*-
"""mk_z19.py —— z19 批次：句末短音裁决 + 女声引擎裁决重投。

## 为什么有这一批
1. **句末短音（新发现，用户原话「嘎哈」）**：xf5272-prod-21 定位到两处
   句末停顿里的 0.2s voiced 短音（20.2s / 22.8s，峰值比全篇均值高 11~17 dB）。
   🔴 按 S6，物理差异 ≠ 听得出，必须先出裁决题验证可闻性再动手修。
2. **z18 作废重投**：z18 收到 2 票，两人控制题都答错（①②本就是同一段音频），
   按规范整轮作废 → 女声换不换 GS 还没有有效结论，需要重投。

## 题目清单（5 题）
| id | 问什么 | 拿到答案改什么 |
|---|---|---|
| blip | 句末短音：原样 vs 压制版 | 选②→可闻，去找根因修；选①/③→不可闻，封存 |
| noise | 女声杂音 GS vs 现役 | 支持/反对换 GS |
| tone | 女声音色 GS vs 现役 | 同上（GS 唯一劣于现役的维度） |
| nat | 女声自然度 GS vs 现役 | 同上 |
| ctrl | 控制题（两段相同） | 选①或② → 本轮作废 |

## 🔴 规范核对
- A 恒为「我们的」（blip 的 A 是现役原样；noise/tone/nat 的 A 是 GS ——
  z18 定下的「A=GS」，保持一致才可比）
- 盲听：选项只写「① 第 1 段」，身份只在服务端 note
- 有「③ 差不多」但没有「不知道」类兜底
- 版本 v1.4，与服务端真相源一致（发布后 curl /api/version 验 consistent）
- 切批次前备份 items.json / votes.jsonl

用法：python3 mk_z19.py
"""
import json
import os
import shutil
import subprocess
import sys
import time

FB = "/mnt/tts/fb"
VER = "v1.4"
ROUND = "z19"


def opt(a_file, a_lab="① 第 1 段", b_file=None, b_lab="② 第 2 段"):
    return [
        {"key": "A", "file": a_file, "label": a_lab},
        {"key": "B", "file": b_file, "label": b_lab},
        {"key": "C", "file": None, "label": "③ 差不多 / 听不出明显区别"},
    ]


ITEMS = [
    {
        "id": "v1.4-z19-blip",
        "dim": "女声·句末短音",
        "title": "这两段，**哪一段听起来更自然**？",
        "desc": "两段读的是同一段话，时长一样、音量一样。\n"
                "**只问一件事：有没有哪一段在「一句话说完、停顿下来」的时候，"
                "突然冒出来一个很短的怪音？**\n"
                "比如像「呃」「啊」这样不属于句子内容的声音，或者一个突兀的短促响声。\n"
                "听一遍就选，凭第一感觉。",
        "ref": None,
        "note": "句末短音可闻性裁决（S6）。\n"
                "①=现役原样（xf5272-prod-21）\n"
                "②=把 20.15~20.64s 与 22.77~23.05s 两处句末短音压到 -42dB 的版本\n"
                "   （12ms 淡变防 click；LUFS 已对齐 -22.90，排除响度干扰）\n"
                "客观：两处短音峰值比全篇均值高 +11.4 / +17.3 dB。\n"
                "选② → 可闻且是缺陷 → 继续找根因修。\n"
                "选①或③ → 听不出 → 该方向封存，别再往里加码。",
        "ver": VER,
        "options": opt("z51_a_s1.wav", b_file="z51_b_s1.wav"),
    },
    {
        "id": "v1.4-z19-noise",
        "dim": "女声·引擎裁决",
        "title": "这两段里，**有没有哪一段能听到杂音**？",
        "desc": "两段读的是同一段话。\n**只问一件事：有没有杂音？**\n"
                "比如沙沙声、嘶嘶声、滋滋声、嗡嗡声，或者音量一阵一阵的波纹感。\n"
                "听一遍就选，凭第一感觉。",
        "ref": None,
        "note": "杂音裁决（z18 重投，素材不变）。\n①=GS 女声 ②=现役 omnivoice。\n"
                "客观：静音段 4-8k 杂音 GS **4.121** vs 现役 **12.952**（EL 11.371）。\n"
                "选② → 杂音可被听出 → 支持换 GS。\n"
                "选① → 与客观相反，需回头查 GS 的 EQ 是否补过头。\n"
                "选③ → 杂音不构成换引擎理由，看 tone/nat。",
        "ver": VER,
        "options": opt("z50_gs_s1.wav", b_file="z50_cur_s1.wav"),
    },
    {
        "id": "v1.4-z19-tone",
        "dim": "女声·引擎裁决",
        "title": "这两段，**哪一段的声音更像「原来那个人」**？",
        "desc": "两段读的是同一段话。\n**只问一件事：哪个声音更像你熟悉的那个播音员本人？**\n"
                "不用管有没有杂音、不用管语速，只比**音色像不像本人**。",
        "ref": None,
        "note": "音色裁决（换一段素材 s2）。\n①=GS 女声 ②=现役 omnivoice。\n"
                "客观：campplus 音色 GS **0.9196** vs 现役 **0.9253**（GS 略低，"
                "是 GS 唯一劣于现役的维度）→ 这题是换引擎的最大不确定项。\n"
                "选① → GS 音色不输/更好 → 强烈支持换 GS。\n"
                "选② → 音色是硬伤，换 GS 要慎重。",
        "ver": VER,
        "options": opt("z50_gs_s2.wav", b_file="z50_cur_s2.wav"),
    },
    {
        "id": "v1.4-z19-nat",
        "dim": "女声·引擎裁决",
        "title": "换一段话再听一次：**哪一段听起来更像真人在说话**？",
        "desc": "两段读的是同一段话（和上一题不同）。\n"
                "**只问一件事：哪个更像真人在说话，而不是机器念稿？**\n"
                "凭整体感觉选，不用分析具体哪里不一样。",
        "ref": None,
        "note": "自然度裁决（第三段素材 s3，避免员工对同一段疲劳）。\n"
                "①=GS 女声 ②=现役 omnivoice。\n"
                "选① → 支持换 GS。选② → 现役更自然。",
        "ver": VER,
        "options": opt("z50_gs_s3.wav", b_file="z50_cur_s3.wav"),
    },
    {
        "id": "v1.4-z19-ctrl",
        "dim": "控制题",
        "title": "这两段，**哪一段听起来更像真人在说话**？",
        "desc": "和前面一样的听法，凭第一感觉选。",
        "ref": None,
        "note": "🔴 控制题：①②是**完全相同**的音频（z51_a_s1 的两份拷贝）。\n"
                "正确答案只有③。选①或② → 说明本轮作答不可靠，整轮作废。\n"
                "（z18 的 2 位同事都选了②，导致整轮作废。）",
        "ver": VER,
        "options": opt("z51_ctrl1.wav", b_file="z51_ctrl2.wav"),
    },
]


def main():
    # 1) 控制题素材：同一段音频放两次
    src = os.path.join(FB, "z51_a_s1.wav")
    for n in ("z51_ctrl1.wav", "z51_ctrl2.wav"):
        dst = os.path.join(FB, n)
        shutil.copyfile(src, dst)
        print("控制素材", n, os.path.getsize(dst))

    # 2) 自检：所有 file 必须存在
    missing = []
    for it in ITEMS:
        for o in it["options"]:
            f = o.get("file")
            if f and not os.path.exists(os.path.join(FB, f)):
                missing.append("%s:%s" % (it["id"], f))
    if missing:
        print("❌ 缺素材:", missing)
        return 1
    print("✅ 素材齐全")

    # 3) md5 撞车检查（A/B 不能是同一段，控制题除外）
    for it in ITEMS:
        fs = [o["file"] for o in it["options"] if o.get("file")]
        if len(fs) == 2:
            h = []
            for f in fs:
                h.append(subprocess.run(["md5sum", os.path.join(FB, f)],
                                        capture_output=True, text=True
                                        ).stdout.split()[0])
            same = h[0] == h[1]
            expect = it["id"].endswith("ctrl")
            flag = "✅ 符合预期" if same == expect else "❌ 不符"
            print("  %-18s md5 相同=%s (期望%s) %s"
                  % (it["id"], same, expect, flag))
            if same != expect:
                return 1

    # 4) 备份旧批次
    ts = time.strftime("%Y%m%d_%H%M%S")
    for f in ("items.json", "votes.jsonl"):
        p = os.path.join(FB, f)
        if os.path.exists(p):
            shutil.copyfile(p, p + ".bak-" + ts)
            print("备份", f, "->", f + ".bak-" + ts)

    # 5) 落盘（先写 items_z19.json，再 cp 到 items.json）
    out = os.path.join(FB, "items_%s.json" % ROUND)
    with open(out, "w", encoding="utf-8") as fp:
        json.dump({"items": ITEMS}, fp, ensure_ascii=False, indent=1)
    shutil.copyfile(out, os.path.join(FB, "items.json"))
    print("写入", out, "并上线 items.json，共", len(ITEMS), "题")
    return 0


if __name__ == "__main__":
    sys.exit(main())
