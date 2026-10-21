#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 z36 轮 **盲听裁决**：数字零填充（v3.6）到底有没有把员工的症状消掉？

## 为什么出这一轮（S6 可闻性验证，规范 5 / z44 教训）
2026-10-21 员工反馈 202610210738（男）/ 202610210939（女）：
「0.1 秒没音」「额/嗯/啊」「电磁音全程很多」「碰碰声全程个别地方有」。
fback53d 取证：员工报的 26 个时间点 **24 个**精确落在 |y|<1e-9 的数字零段上
（男 52 处 / 女 44 处 ≥20ms），EL 标杆同口径 0 处。
v3.6 把这些零段填成 -80dB 环境声，客观清零（L1/L2 已验）。

🔴 但 z44 的教训是：**物理差 ≠ 听得出来**（13dB 改动 9 票仅 1 票）。
所以必须让员工盲听拍板：填完之后，电磁音 / 碰碰声是不是真的没了。

## 素材（A/B 是**同文案 + 同 seed** 的两次生成，逐秒可对齐）
A = v3.6（开了数字零填充，L3z36-*）
B = v3.5（切回 .bak-before-zerofill53 重生成的对照，L3v35c-*）
另有一题 A = v3.6 vs EL 标杆，看离标杆还差多少。

## 🔴 取样位置不是拍脑袋，是**自动探测 B 侧的零段密集区**
规范第 7 节明令禁止「在没有该现象的素材上验证补丁」。所以本脚本先扫 B 侧
（v3.5，有零段的那版）的全篇零段，按 20s 窗计数，取**最密且不相邻的两个窗**
作为取样点 —— 保证题目素材里确实存在员工报的那个现象。

## 每题的决策价值（答不出「拿到答案我改什么」就不出题）
m_noise / f_noise  杂音（电磁音）维度
    选①(=v3.6) → 填充可闻地有效 → v3.6 保留并正式发布
    选②(=v3.5) → v3.6 反而更差 → 回退 v3.6，重新定标填充电平
    选③        → 听不出 → 按 z44 封存：不再往这个方向加码
m_gap / f_gap      卡顿（断裂/碰碰声）维度，同上映射
m_el               v3.6 vs EL 标杆，看剩余差距
ctrl               同一段放两遍；选①或② → 本轮整份作废

用法：TTS_VER=v3.6 TTS_ROUND=z36 venv/bin/python mk_z36.py
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
TMPD = "/mnt/tts/diag/_z36"
TGT_LUFS = -26.1
SR = 24000
DUR = 20.0
ASR_SR = 16000

EL = {"male": "/mnt/tts/data/cases/1a6c20a60481/merged.wav",
      "female": "/mnt/tts/data/cases/00b6a45ad6aa/merged.wav"}


def latest(pat):
    """产物目录会被 _unique_dir 加 -N 后缀，按 mtime 取最新。"""
    c = sorted(glob.glob(pat), key=os.path.getmtime)
    return c[-1] if c else None


A = {"male": latest("/mnt/tts/output/L3z36-male*/L3z36-male-1.mp3"),
     "female": latest("/mnt/tts/output/L3z36-female*/L3z36-female-1.mp3")}
B = {"male": latest("/mnt/tts/output/L3v35c-male*/L3v35c-male-1.mp3"),
     "female": latest("/mnt/tts/output/L3v35c-female*/L3v35c-female-1.mp3")}

VER = os.environ.get("TTS_VER", "v3.6")
ROUND = os.environ.get("TTS_ROUND", "z36")
SIM_MIN = 0.55
SEARCH = (-6, -3, 0, 3, 6)
ASR_MIN_CHARS = 20
WIN = 20.0            # 零段密度统计窗


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


def load16(path):
    out = tempfile.mktemp(suffix=".raw")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                    "-ac", "1", "-ar", str(ASR_SR), "-c:a", "pcm_f32le",
                    "-f", "f32le", out], check=True)
    y = np.fromfile(out, dtype="<f4").astype(np.float32)
    os.remove(out)
    return y


def zero_runs(y, sr=ASR_SR, min_ms=20.0):
    z = np.abs(y) < 1e-9
    d = np.diff(np.concatenate(([0], z.astype(np.int8), [0])))
    s = np.where(d == 1)[0]
    e = np.where(d == -1)[0]
    return [a / sr for a, b in zip(s, e) if (b - a) / sr * 1000.0 >= min_ms]


def dense_windows(path, n=2, dur_limit=None):
    """扫全篇零段，按 WIN 秒分窗计数，返回最密且不相邻的 n 个窗起点。"""
    y = load16(path)
    total = len(y) / ASR_SR
    zr = zero_runs(y)
    if not zr:
        return []
    nw = int(total // WIN)
    cnt = [0] * nw
    for t in zr:
        i = int(t // WIN)
        if 0 <= i < nw:
            cnt[i] += 1
    order = sorted(range(nw), key=lambda i: -cnt[i])
    picked = []
    for i in order:
        if cnt[i] == 0:
            break
        t = i * WIN
        # 留出切片空间，且不和已选窗相邻
        if t + DUR > total - 1.0:
            continue
        if any(abs(t - p) < WIN * 2 for p in picked):
            continue
        picked.append(t)
        if len(picked) >= n:
            break
    print("   B侧全篇 %d 处零段，密度前%d窗（起点/该窗零段数）：%s"
          % (len(zr), n, [(round(p, 1), cnt[int(p // WIN)]) for p in picked]))
    return [p for p in picked if cnt[int(p // WIN)] > 0]


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


def align(src_a, src_b, t_b, dur=DUR, ka=1.0):
    """B 侧 t_b 位置 → A 侧对应位置（ASR 相似度搜偏移）。"""
    hb = hyp(clip(src_b, t_b, dur, tag="B"))
    if len(hb) < ASR_MIN_CHARS:
        return None, len(hb), 0.0
    base = t_b * ka
    best = None
    for off in SEARCH:
        pos = max(base + off, 0.0)
        s = sim(hb, hyp(clip(src_a, pos, dur, tag="A")))
        if best is None or s > best[1]:
            best = (pos, s)
    return (best[0] if best else None), len(hb), (best[1] if best else 0.0)


def emit(wav, name):
    g = TGT_LUFS - lufs(wav)
    out = os.path.join(FB_DIR, "%s.wav" % name)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav,
                    "-af", "volume=%gdB" % g, "-c:a", "pcm_s16le", out],
                   check=True)
    back = lufs(out)
    print("     %-16s md5=%s LUFS %6.1f %s"
          % (name, md5(out), back or 0,
             "OK" if back and abs(back - TGT_LUFS) < 1.0 else "!!偏差大"))
    return out


def two_q(iid, a, b, dim, title, desc, note):
    return {
        "id": iid, "dim": dim, "title": title, "desc": desc,
        "ref": None, "note": note, "ver": VER,
        "options": [
            {"key": "A", "file": a, "label": "① 第 1 段更干净"},
            {"key": "B", "file": b, "label": "② 第 2 段更干净"},
            {"key": "C", "file": None, "label": "③ 两段都干净 / 听不出区别"}]}


NOISE_DESC = ("两段读的是**同一段话**，声音不完全一样是正常的。\n"
              "**只听一件事：有没有杂音 / 电流声（滋滋、沙沙的底噪）**。\n"
              "不用管嗓音像不像、也不用管快慢。\n"
              "听一遍就选，别反复比较。")

GAP_DESC = ("两段读的是**同一段话**。\n"
            "**只听一件事：有没有突然断一下、咔一下、或者中间空一拍**。\n"
            "（就是听着像「卡了」或者有人轻轻「嗯 / 额」了一声的那种感觉）\n"
            "听一遍就选，别反复比较。")


def main():
    os.makedirs(TMPD, exist_ok=True)
    for g in ("male", "female"):
        for k in ("A", "B"):
            p = (A if k == "A" else B)[g]
            if not p or not os.path.exists(p):
                print("❌ 缺素材 %s/%s：%s" % (g, k, p))
                return 1
            print("%s %s = %s" % (g, k, p))

    made, info = {}, {}
    for g in ("male", "female"):
        a, b = A[g], B[g]
        da, db = dur_of(a), dur_of(b)
        ka = da / db
        print("\n=== %s  我们 v3.6 %.1fs / v3.5 %.1fs → K=%.4f ===" % (g, da, db, ka))
        spots = dense_windows(b, n=2)
        if not spots:
            print("❌ %s：v3.5 侧没扫到零段，不能出题（素材无该现象）" % g)
            return 1
        for i, t_b in enumerate(spots):
            slug = ("%s_noise" if i == 0 else "%s_gap") % g[0]
            pos_a, nch, s = align(a, b, t_b, DUR, ka)
            if s < SIM_MIN:
                print("   ❌ %s 位置 %.1fs 对不齐（相似度 %.3f，ASR %d 字）"
                      % (slug, t_b, s, nch))
                return 1
            print("   ✅ %s  v3.5 %.1fs ↔ v3.6 %.1fs  相似度 %.3f"
                  % (slug, t_b, pos_a, s))
            info[slug] = (t_b, pos_a, s)
            made["z36_a_%s" % slug] = emit(clip(a, pos_a, DUR, tag="A"),
                                           "z36_a_%s" % slug)
            made["z36_b_%s" % slug] = emit(clip(b, t_b, DUR, tag="B"),
                                           "z36_b_%s" % slug)

    # ---- v3.6 vs EL 标杆 ----
    print("\n=== m_el（v3.6 vs EL 标杆，中段）===")
    a, e = A["male"], EL["male"]
    da, de = dur_of(a), dur_of(e)
    ke = de / da
    chosen = None
    for t in (300.0, 350.0, 250.0):
        ha = hyp(clip(a, t, DUR, tag="A"))
        if len(ha) < ASR_MIN_CHARS:
            continue
        base = t * ke
        best = max(((max(base + o, 0.0), sim(ha, hyp(clip(e, max(base + o, 0.0),
                                                         DUR, tag="EL"))))
                    for o in SEARCH), key=lambda x: x[1])
        print("   位置 %.1fs → EL %.1fs 相似度 %.3f" % (t, best[0], best[1]))
        if best[1] >= SIM_MIN:
            chosen = (t, best[0], best[1])
            break
    if chosen is None:
        print("❌ m_el 对不齐，不出题")
        return 1
    info["m_el"] = chosen
    made["z36_a_m_el"] = emit(clip(a, chosen[0], DUR, tag="A"), "z36_a_m_el")
    made["z36_b_m_el"] = emit(clip(e, chosen[1], DUR, tag="EL"), "z36_b_m_el")

    # ---- 控制题：同一段放两遍 ----
    print("\n=== 控制题（同一段放两遍）===")
    made["z36_ctrl1"] = emit(clip(A["male"], 60.0, DUR, tag="CTRL"), "z36_ctrl1")
    made["z36_ctrl2"] = emit(clip(A["male"], 60.0, DUR, tag="CTRL"), "z36_ctrl2")

    print("\n=== md5 自检（防素材撞车）===")
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

    DEC = ("\n拿到答案我改什么：\n"
           "  选①（第1段=v3.6 更干净）→ 数字零填充**可闻地有效**，v3.6 保留并正式发布。\n"
           "  选②（第2段=v3.5 更干净）→ v3.6 反而更差，**回退** v3.6，重新定标填充电平。\n"
           "  选③ → 听不出差别：按 z44 教训（物理差≠可闻），**该方向封存**，不再加码。")

    def dec_note(head, spot):
        """note = 题面说明 + 统一决策映射。

        🔴 单独成函数是为了绕开 Python 的坑：相邻字符串字面量后面紧跟
        `(...)` 会被当成**函数调用**；而 `%` 优先级高于 `+`，
        `"a" + DEC % (x)` 会把参数喂给没有格式符的 DEC。
        所以整串先拼好，再一次格式化。
        """
        return (head + DEC) % (spot[0], spot[1], spot[2])

    items = [
        two_q(qid("m_noise"), "z36_a_m_noise.wav", "z36_b_m_noise.wav",
              "男声·杂音（电流声）",
              "这两段，**哪一段杂音 / 电流声更少**？",
              NOISE_DESC,
              dec_note("男声杂音维度。取样点 = v3.5 侧**零段最密集**的 20s 窗"
                       "（v3.5 %.1fs ↔ v3.6 %.1fs，ASR 相似度 %.2f）——\n"
                       "只有取在「有该现象」的位置上，题目才有判别力"
                       "（规范 7：禁止在无现象素材上验证补丁）。\n"
                       "客观：该窗 v3.5 有零段 / v3.6 已填平"
                       "（全篇 52 处 → 0 处）。", info["m_noise"])),

        two_q(qid("m_gap"), "z36_a_m_gap.wav", "z36_b_m_gap.wav",
              "男声·卡顿（断裂 / 咔哒）",
              "这两段，**哪一段听着更顺、更少卡顿**？",
              GAP_DESC,
              dec_note("男声卡顿维度（员工原话「碰碰声」「0.1 秒没音」）。\n"
                       "取样点 = v3.5 侧零段**第二密集**的窗"
                       "（v3.5 %.1fs ↔ v3.6 %.1fs，相似度 %.2f），\n"
                       "换位置复验，排除「只有一处碰巧」。\n"
                       "客观：v3.5 该处 30ms 内陡降 ≥25dB；"
                       "v3.6 全篇陡降 169 → 42（EL 28）。", info["m_gap"])),

        two_q(qid("f_noise"), "z36_a_f_noise.wav", "z36_b_f_noise.wav",
              "女声·杂音（电流声）",
              "这两段，**哪一段杂音 / 电流声更少**？",
              NOISE_DESC,
              dec_note("女声杂音维度（员工原话「电磁音全程很多」）。\n"
                       "取样点 = v3.5 侧零段最密集窗"
                       "（v3.5 %.1fs ↔ v3.6 %.1fs，相似度 %.2f）。\n"
                       "客观：女声全篇零段 44 → 0 处。", info["f_noise"])),

        two_q(qid("f_gap"), "z36_a_f_gap.wav", "z36_b_f_gap.wav",
              "女声·卡顿（断裂 / 咔哒）",
              "这两段，**哪一段听着更顺、更少卡顿**？",
              GAP_DESC,
              dec_note("女声卡顿维度（员工报的「嗯 / 额 / 啊」逐秒点位）。\n"
                       "取样点 = v3.5 侧零段第二密集窗"
                       "（v3.5 %.1fs ↔ v3.6 %.1fs，相似度 %.2f）。\n"
                       "客观：女声陡降 45 → 0~5（EL 0）。", info["f_gap"])),

        two_q(qid("m_el"), "z36_a_m_el.wav", "z36_b_m_el.wav",
              "男声·杂音再测（换一段）",
              "这两段，**哪一段杂音 / 电流声更少**？",
              NOISE_DESC,
              "🔴 这一题不是前后对照，是**看离标杆还差多少**"
              "（v3.6 %.1fs ↔ 标杆 %.1fs，相似度 %.2f）。\n"
              "①=v3.6 ②=标杆。\n"
              "选① → 我们已经不比标杆吵 → 这个维度可以收工。\n"
              "选② → 仍有可闻差距 → 继续定位（但先在 v3.5/v3.6 对照题上确认方向对不对）。\n"
              "选③ → 该维度已对齐标杆。"
              % (info["m_el"][0], info["m_el"][1], info["m_el"][2])),

        two_q(qid("ctrl"), "z36_ctrl1.wav", "z36_ctrl2.wav",
              "控制题",
              "这两段，**哪一段杂音 / 电流声更少**？",
              "听一遍就选，凭第一感觉。",
              "🔴 **控制题：①和②是同一段音频**（v3.6 男声 60s 处 20s）。\n"
              "若多数人选①或② → 本轮存在瞎选 / 位置偏好，**整份作废**需重出。\n"
              "选③ = 判别力正常，本轮结论可用。"),
    ]

    print("\n=== 盲听自检 ===")
    banned = ("我们的", "我们", "EL", "标杆", "参考", "标准", "v3.6", "v3.5",
              "omnivoice", "declick", "零段", "填充", "数字零", "click")
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

    out = os.environ.get("TTS_ITEMS_OUT", os.path.join(FB_DIR, "items_z36.json"))
    json.dump({"items": items}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n已写入 %s（%d 题，ver=%s / round=%s）" % (out, len(items), VER, ROUND))
    chk = json.load(open(out, encoding="utf-8"))
    assert isinstance(chk, dict) and "items" in chk
    print("  ✅ 格式自检：dict.items 有 %d 题" % len(chk["items"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
