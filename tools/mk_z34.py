#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 z34 轮 **盲听验收**：v3.5 链路能不能替代 EL？

## 为什么出这一轮
v3.1~v3.5 这五轮改动（clickfix / 窄带 EQ / 女声链接管 / headfix 文件头孤立段
切除 / headfix 判据收紧 + numcheck53）**全部只有客观指标验证，从未盲听验收过**。
而项目目标就是「员工听不出我们和 EL 的差别」—— 这条只能由盲听拍板
（🔴 z20 教训：GS 女声客观全面优于 EL，员工一致判「明显是机器」）。

更关键：规范里记着「男声基线那 22 票全 EL」是**在标了身份的口径下拿的**，
含金量存疑，必须盲听重测。本轮就是那次重测。

## 素材
A = 我们 v3.5 长篇产物（案例库长文案，生产 API 直出，走完整后处理链）
B = EL 标杆 cases/<id>/merged.wav（同文案）

🔴 跨成品 A/B 必须先按文本对齐（规范 5.10）：两条时长不同
（男 495.4s vs EL 483.8s），线性换算会累积偏移 → 用 ASR 搜最佳起点，
相似度 ≥0.55 才出題。

## 每题的决策价值（答不出「拿到答案我改什么」就不出题）
m_nat / m_nat2  男声盲听重测（两个位置，防单段偶然）
                选② → 还能听出 EL 更好 → 继续定位；选③ → 男声验收通过
f_nat           女声盲听（中段）
f_head          女声起音专项（v3.4/v3.5 修的文件头孤立段就是这个维度）
                选② → 起音仍不如 EL → 继续做 #7 型多组崩坏；
                选③ → headfix 已够，关闭 #7 这条线
ctrl            控制题：同一段放两遍，选①或② → 整份作废

用法：TTS_VER=v3.5 TTS_ROUND=z34 venv/bin/python mk_z34.py
"""
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")
from asr_check import normalize, transcribe  # noqa: E402

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
TMPD = "/mnt/tts/diag/_z34"
TGT_LUFS = -26.1
SR = 24000
DUR = 20.0
DUR_HEAD = 8.0

# A（我们 v3.5）/ B（EL 标杆）
PAIRS = [
    # slug, 性别, 我们产物, EL 标杆, 位置, 时长
    ("m_nat",  "male",   "/mnt/tts/output/L3v35-male/L3v35-male-1.mp3",
     "/mnt/tts/data/cases/1a6c20a60481/merged.wav", 390.0, DUR),
    ("m_nat2", "male",   "/mnt/tts/output/L3v35-male/L3v35-male-1.mp3",
     "/mnt/tts/data/cases/1a6c20a60481/merged.wav", 150.0, DUR),
    ("f_nat",  "female", "/mnt/tts/output/L3v35-female/L3v35-female-1.mp3",
     "/mnt/tts/data/cases/00b6a45ad6aa/merged.wav", 390.0, DUR),
    ("f_head", "female", "/mnt/tts/output/L3v35-female/L3v35-female-1.mp3",
     "/mnt/tts/data/cases/00b6a45ad6aa/merged.wav", 0.0, DUR_HEAD),
]
CTRL = ("/mnt/tts/output/L3v35-male/L3v35-male-1.mp3", 60.0, DUR_HEAD)

VER = os.environ.get("TTS_VER", "v3.5")
ROUND = os.environ.get("TTS_ROUND", "z34")
SIM_MIN = 0.55
SEARCH = (-8, -4, 0, 4, 8)          # B 起点偏移搜索（秒）


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
        sys.exit("❌ 切片越界：%s 第 %.1fs 起只有 %.1fs" % (tag or src, s0, got))
    return t


def hyp(wav):
    return normalize(transcribe(wav, language="Korean"))


def sim(a, b):
    import difflib
    return difflib.SequenceMatcher(None, a[:120], b[:120], autojunk=False).ratio()


def emit(wav, name):
    """输出 wav（fb 只按文件名取，不编码 mp3，排除编码变量）。"""
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


def two_q(iid, a, b, dim, title, desc, note):
    return {
        "id": iid, "dim": dim, "title": title, "desc": desc,
        "ref": None, "note": note, "ver": VER,
        "options": [
            {"key": "A", "file": a, "label": "① 第 1 段"},
            {"key": "B", "file": b, "label": "② 第 2 段"},
            {"key": "C", "file": None, "label": "③ 差不多 / 听不出明显区别"}]}


def main():
    os.makedirs(TMPD, exist_ok=True)
    for _, _, a, b, _, _ in PAIRS:
        for p in (a, b):
            if not os.path.exists(p):
                print("❌ 缺素材：%s" % p)
                return 1

    made = {}
    info = {}
    for slug, gender, ours, el, t, d in PAIRS:
        du, de = dur_of(ours), dur_of(el)
        K = de / du
        print("=== %s（%s） 我们 %.1fs / EL %.1fs → K=%.4f ==="
              % (slug, gender, du, de, K))
        ca = clip(ours, t, d, tag="OURS")
        ha = hyp(ca)
        base = t * K
        cands = []
        for off in SEARCH:
            pos = max(base + off, 0.0)
            if any(abs(pos - p) < 0.05 for p, _ in cands):
                continue
            cands.append((pos, off))
        best = None
        for pos, off in cands:
            s = sim(ha, hyp(clip(el, pos, d, tag="EL")))
            print("     EL 起点 %.1fs (偏移 %+d)  相似度 %.3f" % (pos, off, s))
            if best is None or s > best[1]:
                best = (pos, s)
        print("   → 采用 EL %.1fs，相似度 %.3f %s\n"
              % (best[0], best[1], "✅" if best[1] >= SIM_MIN else "❌"))
        if best[1] < SIM_MIN:
            print("❌ 相似度不足 %.2f，不出题" % SIM_MIN)
            return 1
        info[slug] = (t, best[0], best[1], du, de)
        made["z34_a_%s" % slug] = emit(clip(ours, t, d, tag="OURS"),
                                       "z34_a_%s" % slug)
        made["z34_b_%s" % slug] = emit(clip(el, best[0], d, tag="EL"),
                                       "z34_b_%s" % slug)

    print("=== 控制题（同一段放两遍）===")
    cs, ct, cd = CTRL
    made["z34_ctrl1"] = emit(clip(cs, ct, cd, tag="CTRL"), "z34_ctrl1")
    made["z34_ctrl2"] = emit(clip(cs, ct, cd, tag="CTRL"), "z34_ctrl2")

    print("\n=== md5 自检（防不小心两份素材相同）===")
    inv = {}
    for nm, p in made.items():
        inv.setdefault(md5(p), []).append(nm)
    bad = False
    for h, ns in inv.items():
        if len(ns) > 1 and "ctrl" not in ns[0]:
            print("  ❌ 撞车 %s -> %s" % (h, ns))
            bad = True
    if not bad:
        print("  ✅ 无撞车（ctrl1/ctrl2 刻意相同）")

    NAT_DESC = ("两段读的是**同一段话**，声音不完全一样是正常的。\n"
                "**只问一件事：哪一段更像真人在自然地说话？**\n"
                "不用管快慢、也不管嗓音像不像谁，就凭整体感觉。\n"
                "听一遍就选，别反复比较。")

    items = [
        two_q(qid("m_nat"), "z34_a_m_nat.wav", "z34_b_m_nat.wav",
              "男声·整体自然度（盲听）",
              "这两段，**哪一段听起来更像真人在说话**？",
              NAT_DESC,
              "🔴 **本轮核心：男声盲听重测**（中段，我们 %.1fs / EL %.1fs，"
              "ASR 相似度 %.2f）。\n"
              "①=我们 v3.5  ②=EL 标杆。\n"
              "客观（v3.5 男声）：click 0.09×(EL 为 1)、中频 1.01、质心 0.99、"
              "杂音 0.77、cps 8.27(EL 8.37)、音色 0.9329。\n"
              "选② → 员工仍能听出 EL 更好 → 客观对齐≠听感对齐，\n"
              "       回 S1 重新定位（别再只盯频谱指标）。\n"
              "选③ → 听不出差别 → **男声验收通过**。\n"
              "选① → 我们被判更像真人 → 验收通过且优于 EL。"
              % (info["m_nat"][0], info["m_nat"][1], info["m_nat"][2])),

        two_q(qid("m_nat2"), "z34_a_m_nat2.wav", "z34_b_m_nat2.wav",
              "男声·整体自然度（换一段复验）",
              "换一段话再听一次：**哪一段听起来更像真人在说话**？",
              "还是同样的问法，换了一段内容（前段）。\n听一遍就选，凭第一感觉。",
              "男声复验（前段，我们 %.1fs / EL %.1fs，相似度 %.2f）。\n"
              "换位置复验，排除「只有中段碰巧好听」。\n"
              "若 m_nat 与 m_nat2 结论一致 → 裁决成立；\n"
              "若相反 → 差异不稳定，本轮不作废但不作为验收依据。"
              % (info["m_nat2"][0], info["m_nat2"][1], info["m_nat2"][2])),

        two_q(qid("f_nat"), "z34_a_f_nat.wav", "z34_b_f_nat.wav",
              "女声·整体自然度（盲听）",
              "这两段，**哪一段听起来更像真人在说话**？",
              NAT_DESC,
              "女声盲听（中段，我们 %.1fs / EL %.1fs，ASR 相似度 %.2f）。\n"
              "①=我们 v3.5  ②=EL 标杆。\n"
              "客观（v3.5 女声）：click 0.00×、中频 1.01、质心 0.99、"
              "杂音 1.12、cps 1.00、音色 0.9227。\n"
              "选② → 女声还有可闻差距 → 回 S1 定位（旧「女声 0.8172 最大短板」"
              "已作废，别再信）。\n"
              "选③ → 女声验收通过。"
              % (info["f_nat"][0], info["f_nat"][1], info["f_nat"][2])),

        two_q(qid("f_head"), "z34_a_f_head.wav", "z34_b_f_head.wav",
              "女声·起音专项（盲听）",
              "只听**开头这几秒**：哪一段的开头更干净、更自然？",
              "两段读的是同一句话的开头。\n"
              "**只听第一句**：有没有「还没正式开口就先冒出来一个音」、\n"
              "或者开头拖一下、含混一下的感觉？\n"
              "听一遍就选，凭第一感觉。",
              "🔴 **起音专项 = v3.4/v3.5 修的那个维度**（文件头孤立段：\n"
              "起音先抛 100-200ms 垃圾碎片 → 200-600ms 近静音 → 才进正文）。\n"
              "⚠️ 本标杆文案开头是 아모텍，**不触发**该缺陷（触发需外语专有名词\n"
              "如 로보티즈 开头），所以本题是「起音整体是否对齐 EL」的对照，\n"
              "不是修复前后对照。\n"
              "选② → 起音仍不如 EL → 继续做 #7 型多组崩坏（占产出 9%）。\n"
              "选③ → 起音已对齐，headfix 足够，**关闭 #7 这条线**。"),

        two_q(qid("ctrl"), "z34_ctrl1.wav", "z34_ctrl2.wav",
              "控制题",
              "这两段，**哪一段听起来更像真人在说话**？",
              "听一遍就选，凭第一感觉。",
              "🔴 **控制题：①和②是同一段音频**（我们男声 60s 处 8s）。\n"
              "若多数人选①或② → 说明本轮存在瞎选/位置偏好，\n"
              "**本轮所有结论作废**，需要重出。\n"
              "选③ = 判别力正常，本轮结论可用。"),
    ]

    print("\n=== 盲听自检 ===")
    banned = ("我们的", "我们", "EL", "标杆", "参考", "标准", "v3.5", "v3.4",
              "omnivoice", "headfix", "click", "音色", "아모텍", "로보티즈")
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
            print("  ❌ 泄题 %s %s 含 %r" % x)
        return 1
    print("  ✅ 题面与选项无身份词（dim 也查了，它同样显示在卡片上）")

    out = os.environ.get("TTS_ITEMS_OUT", os.path.join(FB_DIR, "items_z34.json"))
    json.dump({"items": items}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n已写入 %s（%d 题，ver=%s / round=%s）" % (out, len(items), VER, ROUND))

    chk = json.load(open(out, encoding="utf-8"))
    assert isinstance(chk, dict) and "items" in chk
    print("  ✅ 格式自检：dict.items 有 %d 题" % len(chk["items"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
