#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第48轮 **z18 女声引擎裁决**：女声要不要换成 GS？

## 为什么出这一轮
v1.3 定案了 GS 女声的固定 EQ（`GS_EQ_F`），成品 `gs48f_eq` 五个频段全部贴到 EL：

| 指标 | EL 标杆 | 现役 omnivoice v1.1 | **GS v1.3** |
|---|---|---|---|
| 音色余弦 | 1.000 | **0.9253** | 0.9196（−0.006） |
| click % | 0.157 | 0.229 | **0.140** |
| 静音 4-8k 杂音 | 11.371 | 12.952 | **4.121** |
| cps | 7.45 | 7.53 | **7.450** |
| CER | — | 0.0393 | 0.0527 |

杂音 / click 全方位优于现役，**唯独音色低 0.006**。
「能不能换」是 A 级变更（换引擎 + 换参考音频），**必须员工裁决**，
不能只看数字（🔴 z44 教训：13dB 的物理改动，9 票只有 1 票指向修后）。

## 🔴 与 z16 的关系：z16 被本轮吸收
z16 问的是「我们 vs EL」，z18 问的是「GS vs 现役」。
两者的杂音/音色/自然度维度重合，而 z18 直接给出**换不换引擎**的决策依据，
信息量严格更大。z16 保留题库文件但不优先发放。

## 素材对齐（跨成品 A/B 铁律）
两条成品不同引擎、**时长也不同**（GS 458.4s / 现役 453.2s），
线性比例换算会累积偏移 → 用 ASR 在估算位置附近**搜索最佳起点**，
相似度 ≥0.55 才算同一句，否则不产出。

用法：TTS_VER=v1.4 venv/bin/python mk_z50.py
"""
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")
from asr_check import normalize, transcribe  # noqa: E402

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
TMPD = "/mnt/tts/diag/_z50"
TGT_LUFS = -26.1
SR = 24000
DUR = 20.0

# A = GS 女声 v1.3 定案；B = 现役 omnivoice v1.1
GS = "/mnt/tts/output/gs48f_eq/gs48f_eq-1.mp3"
CUR = "/mnt/tts/output/v11_f_official/v11_f_official-1.mp3"

VER = os.environ.get("TTS_VER", "v1.4")
ROUND = os.environ.get("TTS_ROUND", "z18")
SIM_MIN = 0.55
SEARCH = (-6, -3, 0, 3, 6)          # B 起点偏移搜索（秒）


def qid(slug):
    return "%s-%s-%s" % (VER, ROUND, slug)


def lufs(p):
    r = subprocess.run(["ffmpeg", "-i", p, "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    v = [l for l in r.stderr.splitlines() if "I:" in l and "LUFS" in l]
    return float(v[-1].split("I:")[1].split("LUFS")[0].strip()) if v else None


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()[:8]


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
    print("     %-12s md5=%s LUFS %6.1f %s"
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
    for p in (GS, CUR):
        if not os.path.exists(p):
            print("❌ 缺素材：%s" % p)
            return 1
    dur_gs = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", GS], capture_output=True,
        text=True).stdout.strip())
    dur_cur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", CUR], capture_output=True,
        text=True).stdout.strip())
    K = dur_cur / dur_gs
    print("时长：GS %.1fs / 现役 %.1fs → 线性换算 K=%.5f" % (dur_gs, dur_cur, K))
    print("🔴 但不同引擎语速不同，线性换算会累积偏移 → 用 ASR 搜最佳起点\n")

    pairs = []
    for i, t in enumerate([0.0, 120.0, 300.0], start=1):
        print("=== 位置 s%d @ GS %.0fs ===" % (i, t))
        ga = clip(GS, t, tag="GS")
        ha = hyp(ga)
        base = t * K
        # 🔴 候选起点必须先钳到 >=0 再**去重**：clip() 内部有 max(s0,0)，
        #    负偏移会被钳成同一段，导致多个候选切出完全相同的结果、
        #    相似度出现假平局，脚本会错挑第一个并把错误位置写进 note
        #    （实测 GS 0s 处算出「现役 -6.0s」）。
        cands = []
        for off in SEARCH:
            pos = max(base + off, 0.0)
            if any(abs(pos - p) < 0.05 for p, _ in cands):
                continue
            cands.append((pos, off))
        best = None
        for pos, off in cands:
            cb = clip(CUR, pos, tag="CUR")
            s = sim(ha, hyp(cb))
            print("     现役起点 %.1fs (偏移 %+d)  相似度 %.3f" % (pos, off, s))
            if best is None or s > best[1]:
                best = (pos, s)
        print("   → 采用现役 %.1fs，相似度 %.3f %s\n"
              % (best[0], best[1], "✅" if best[1] >= SIM_MIN else "❌"))
        if best[1] < SIM_MIN:
            print("❌ 相似度不足 %.2f，不出题（换个位置重来）" % SIM_MIN)
            return 1
        pairs.append((i, t, best[0], best[1]))

    made = {}
    print("=== 输出 wav + 响度归一 ===")
    for i, t, tb, _ in pairs:
        made["z50_gs_s%d" % i] = emit(clip(GS, t, tag="GS"), "z50_gs_s%d" % i)
        made["z50_cur_s%d" % i] = emit(clip(CUR, tb, tag="CUR"),
                                       "z50_cur_s%d" % i)
    # 控制题：现役 60s 处同一段放两次
    made["z50_ctrl1"] = emit(clip(CUR, 60.0, 6.0, tag="CUR"), "z50_ctrl1")
    made["z50_ctrl2"] = emit(clip(CUR, 60.0, 6.0, tag="CUR"), "z50_ctrl2")

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

    DIM = "女声·引擎裁决"
    items = [
        two_q(qid("noise"), "z50_gs_s1.wav", "z50_cur_s1.wav", DIM,
              "这两段里，**有没有哪一段能听到杂音**？",
              "两段读的是同一段话。\n"
              "**只问一件事：有没有杂音？**\n"
              "比如沙沙声、嘶嘶声、滋滋声、嗡嗡声，或者音量一阵一阵的波纹感。\n"
              "听一遍就选，凭第一感觉。",
              "杂音裁决（开头 s1，GS %.1fs / 现役 %.1fs，ASR 相似度 %.2f）。\n"
              "①=GS 女声 ②=现役 omnivoice。\n"
              "客观：静音段 4-8k 杂音 GS **4.121** vs 现役 **12.952**"
              "（EL 11.371）—— GS 好一个量级。\n"
              "选② → 杂音可被听出 → **支持换 GS**（现役杂音确实更重）。\n"
              "选① → GS 反而被指有杂音 → 与客观相反，需回头查 GS 的 EQ 是不是补过头。\n"
              "选③ → 杂音不构成换引擎的理由，看音色题（tone）裁决。"
              % (pairs[0][1], pairs[0][2], pairs[0][3])),

        two_q(qid("tone"), "z50_gs_s2.wav", "z50_cur_s2.wav", DIM,
              "这两段，**哪一段的声音更像「原来那个人」**？",
              "两段读的是同一段话，声音不一样是正常的。\n"
              "**只问一件事：哪一段的音色更像原来那个人？**\n"
              "不用管快慢和杂音，只听声音本身像不像。\n"
              "听一遍就选，凭第一感觉。",
              "🔴 **本题是换不换 GS 的最大不确定项**（s2 中段）。\n"
              "客观：音色余弦 GS **0.9196** vs 现役 **0.9253**"
              "—— GS **低 0.006**，是唯一劣于现役的指标。\n"
              "选① → GS 也像 → 音色不阻碍，结合杂音/click 优势 → **可以换 GS**。\n"
              "选② → 员工觉得现役更像 → 音色是硬伤，**暂缓换**，\n"
              "       回 S3 继续查 GS 音色缺口（变调吃掉共振峰那条线）。\n"
              "选③ → 音色不可分辨，本项封存。"),

        two_q(qid("nat"), "z50_gs_s2.wav", "z50_cur_s2.wav", DIM,
              "这两段，**哪一段听起来更像真人在说话**？",
              "**只问整体感觉**：哪一段更像真人在自然地说话，\n"
              "哪一段更像机器在念稿？\n"
              "听一遍就选，凭第一感觉。",
              "整体自然度（s2，与 tone 同一对素材，问法不同）。\n"
              "🔴 这是**终极验收口径**——员工验收标准是「听不出差别」。\n"
              "选③ → 听不出差别 → **换哪个都行**，优先选杂音更低的。\n"
              "若前两题都选③而本题选② → 说明还有没量到的维度，\n"
              "需要回 S1 重新定位（别再只盯着频谱指标）。"),

        two_q(qid("nat2"), "z50_gs_s3.wav", "z50_cur_s3.wav", DIM,
              "换一段话再听一次：**哪一段听起来更像真人在说话**？",
              "还是同样的问法，换了一段内容（后段）。\n"
              "听一遍就选，凭第一感觉。",
              "自然度复验（s3 后段，GS %.1fs / 现役 %.1fs，相似度 %.2f）。\n"
              "换位置复验，排除「只有中段碰巧好听」。\n"
              "若 nat 与 nat2 结论一致 → 裁决成立；\n"
              "若相反 → 说明差异不稳定，本轮不作废但不作为换引擎依据。"
              % (pairs[2][1], pairs[2][2], pairs[2][3])),

        two_q(qid("ctrl"), "z50_ctrl1.wav", "z50_ctrl2.wav", DIM,
              "这两段，**哪一段听起来更像真人在说话**？",
              "听一遍就选，凭第一感觉。",
              "🔴 **控制题：①和②是同一段音频**（现役 omnivoice，60s 处 6s）。\n"
              "若多数人选①或② → 说明本轮存在瞎选/位置偏好，\n"
              "**本轮所有结论作废**，需要重出。\n"
              "选③ = 判别力正常，本轮结论可用。"),
    ]

    print("\n=== 盲听自检 ===")
    banned = ("我们的", "我们", "EL", "标杆", "参考", "标准", "GS",
              "v1.1", "v1.3", "v1.4", "omnivoice")
    leak = []
    for it in items:
        if "-ctrl" in it["id"]:
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
            print("  ❌ 泄题 %s %s 含 %r" % x)
        return 1
    print("  ✅ 题面与选项无身份词")

    out = os.environ.get("TTS_ITEMS_OUT", os.path.join(FB_DIR, "items_z18.json"))
    json.dump({"items": items}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n已写入 %s（%d 题，ver=%s）" % (out, len(items), VER))
    print("  ⚠️ 未覆盖线上 items.json。切换批次：cp %s %s/items.json"
          % (out, FB_DIR))

    chk = json.load(open(out, encoding="utf-8"))
    assert isinstance(chk, dict) and "items" in chk
    print("  ✅ 格式自检：dict.items 有 %d 题" % len(chk["items"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
