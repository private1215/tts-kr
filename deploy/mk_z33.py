#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第三十三轮题库：**别再猜了，让员工把「电子音」的时间点标出来**

## 为什么换打法
z 轮 3 人全员答「电子音/吃字/卡字明显还有」（3/3），且口头补了一句
「**挺接近，还是有电子音**」。而我们连续排除了 6 个维度，全部不是主因：

| 假设 | 结果 |
|---|---|
| 频谱能量分布 | ❌ Σz² 19.6，落进 EL 自身分布（17~240） |
| 8k 以上带宽被切 | ❌ EL 自己 8k~12k 也只有 0.00107、12k 以上为 0 |
| 缺噪声细节 | ❌ z4 有 2/3 说「两段都很电子」 |
| 揉两遍 WSOLA | ❌ 已修，且 z1 里员工听不出新旧差别 |
| 吃字（可懂度） | ❌ 滑窗对齐后 CER 我们 0.103 / EL 0.106（0.97x） |
| jitter/shimmer 微扰缺失 | ❌ 我们 0.0484/0.515 vs EL 0.0500/0.541，几乎一样 |

→ 继续猜是浪费员工的耳朵。**这轮改成「定点标注」**：
员工听到问题就暂停，前端 `mode: diagnostic` 会把 `原音频秒=X.XXX`
自动写进文字框，我们拿到秒数后直接分析那一段波形。

## 前端机制（feedback.html:177-219，历史上从未用过）
- 需要 `mode:"diagnostic"` + `ref.file`（播放器取该题第一个 audio）
- `focus.start_seconds` → 「跳到建议起点」按钮
- `version` / `source_sha256` → 写进时间戳里，便于回查版本
- 点「记录当前问题时间点」→ 往全局 `#free` 追加
  `[<id> | <version> | 原音频秒=X.XXX | sha256=...] 现象：`
- 员工在「现象：」后补写「电子音 / 吃字 / 卡字」

🔴 注意：录的是**切片内**的时间。本轮切片统一从成品 0 秒起取 90 秒，
   所以员工记的秒数 = 成品真实秒数，不用加偏移。
"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
TMPD = "/mnt/tts/diag/_z33"
TGT_LUFS = -26.1
DUR = 90.0            # 🔴 比常规 30s 长：给员工足够素材去找问题点
START = 0.0           # 🔴 从 0 秒起，员工记的秒数即成品真实秒数
SR_CLIP = 24000

OURS = "/mnt/tts/output/longprobe-z32-6cdd/longprobe-z32-6cdd-1.mp3"
EL_S = "/mnt/tts/data/cases/6cdd25bc75f2/merged.wav"
OURS_VER = "z32-h5"
EL_VER = "el-6cdd"
SUFFIX = "20261016z2"


def lufs(p):
    r = subprocess.run(["ffmpeg", "-i", p, "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    v = [l for l in r.stderr.splitlines() if "I:" in l and "LUFS" in l]
    return float(v[-1].split("I:")[1].split("LUFS")[0].strip()) if v else None


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()[:8]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cut(src, start, dur, tag):
    t = os.path.join(TMPD, "_%s.wav" % tag)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-ss", "%.1f" % start, "-ar", str(SR_CLIP), "-ac", "1",
                    "-t", "%.1f" % dur, t], check=True)
    got = os.path.getsize(t) / 2.0 / SR_CLIP
    if got < dur - 0.5:
        sys.exit("切片越界：%s 第%.0fs 起取%.0fs，实际仅%.1fs"
                 % (os.path.basename(src), start, dur, got))
    return t


def emit(wav, name, title):
    """volume=%gdB 不是 %gdBB（多一个 B = +18dB）。"""
    g = TGT_LUFS - lufs(wav)
    out = os.path.join(FB_DIR, "%s.mp3" % name)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav,
                    "-af", "volume=%gdB" % g, "-b:a", "192k",
                    "-metadata", "title=%s" % title, out], check=True)
    back = lufs(out)
    print("  %-16s md5=%s  LUFS %6.1f %s"
          % (name, md5(out), back if back else 0,
             "OK" if back and abs(back - TGT_LUFS) < 1.0 else "!!爆表"))
    return out


def mark_item(iid, ver, sha, clipfile, who, desc_extra):
    return {
        "id": iid, "mode": "diagnostic", "version": ver,
        "dim": "电子音/吃字/卡字 定点标注（%s）" % who,
        "title": "【%s】听到问题就暂停，点「记录当前问题时间点」" % who,
        "desc": ("播放下面这段 %d 秒音频（**从成品最开头开始**，你记录的时间"
                 "就是成品里的真实秒数）。\n\n"
                 "听到任意一种问题就**立刻暂停**，然后点下面的\n"
                 "**「记录当前问题时间点」**按钮 —— 时间点会自动填到页面底部"
                 "的反馈框里，**请在那行冒号后面写上是什么现象**"
                 "（电子音 / 吃字 / 卡字，也可以写别的词）。\n\n"
                 "**可以标多个点**，每标一个就再补一行现象。\n"
                 "如果整段都明显有问题，标 1~2 个代表点即可。\n\n"
                 "%s" % (int(DUR), desc_extra)),
        "ref": {"label": "待听：%s（%d 秒，从成品 0 秒起）" % (who, int(DUR)),
                "file": clipfile, "note": ""},
        "focus": {"start_seconds": 0.0},
        "source_sha256": sha,
        "note": ("拿到时间点后的动作：\n"
                 "  按秒数切出那一段波形，逐帧看频谱/包络/谐波结构，\n"
                 "  定位「电子音」到底是哪一个物理现象。\n"
                 "已排除：频段能量、8k 以上带宽、噪声细节、揉两遍 WSOLA、\n"
                 "        CER 可懂度、jitter/shimmer 微扰。"),
        "options": [
            {"key": "A", "file": None, "label": "① 整段都电子"},
            {"key": "B", "file": None, "label": "② 只有某些词/某几句有问题（已标时间点）"},
            {"key": "C", "file": None, "label": "③ 这段没听出问题"}]}


def main():
    for p in (OURS, EL_S):
        if not os.path.exists(p):
            print("缺素材：%s" % p)
            return 1
    os.makedirs(TMPD, exist_ok=True)
    made = {}

    made["z33_mk_ours"] = emit(cut(OURS, START, DUR, "ours"),
                               "z33_mk_ours", "mark ours")
    made["z33_mk_el"] = emit(cut(EL_S, START, DUR, "el"),
                             "z33_mk_el", "mark el")
    ctrl = cut(OURS, START, 20.0, "ctrl")
    made["z33_ctrl1"] = emit(ctrl, "z33_ctrl1", "ctrl A")
    made["z33_ctrl2"] = emit(ctrl, "z33_ctrl2", "ctrl B")

    print("\n=== md5 自检 ===")
    inv = {}
    for nm, p in made.items():
        inv.setdefault(md5(p), []).append(nm)
    bad = False
    for h, ns in inv.items():
        if len(ns) > 1:
            ok = set(ns) <= {"z33_ctrl1", "z33_ctrl2"}
            print("  %-9s %s" % (h, "OK(控制题同素材)" if ok else "撞车 -> %s" % ns))
            if not ok:
                bad = True
    if not bad:
        print("  无撞车")

    items = [
        mark_item("d1_mark_ours_%s" % SUFFIX, OURS_VER, sha256(OURS),
                  "z33_mk_ours.mp3", "我们的当前线上版本",
                  "这是**我们的**声音（男声）。请重点标「电子音」——"
                  "就是那种发假、像机器合成的感觉。"),
        mark_item("d2_mark_el_%s" % SUFFIX, EL_VER, sha256(EL_S),
                  "z33_mk_el.mp3", "EL 对照",
                  "这是 **EL** 的声音（同一文案）。**也请同样标注** —— "
                  "这一步很关键：如果 EL 上你也能标出问题点，说明问题在听音方式或素材；"
                  "如果**只有我们的**标得出，那就确定是我们这边的问题。"),
        {"id": "ctrl_same_%s" % SUFFIX, "dim": "答题有效性校验",
         "title": "【最后一题】这两段，听起来有区别吗？",
         "desc": "**这道题必须选一个**（选错也没关系）。\n\n"
                 "这两段是**同一段音频放了两遍**，内容完全一样。听 5 秒就够。",
         "ref": None,
         "note": "隐藏控制题：正解=C 两段一样。\n"
                 "沿用问法「这两段听起来有区别吗」——已连续 4 轮全对；\n"
                 "旧问法「哪段念得更快」4 人里 2 人错，别再用。",
         "options": [
             {"key": "A", "file": "z33_ctrl1.mp3", "label": "A 明显有区别"},
             {"key": "B", "file": "z33_ctrl2.mp3", "label": "B 明显有区别"},
             {"key": "C", "file": None, "label": "C 没有区别，是一样的"}]},
    ]

    print("\n=== 题目自检 ===")
    ok = True
    for it in items:
        n = sum(1 for o in it["options"] if o.get("file"))
        r = (it.get("ref") or {}).get("file")
        if it.get("mode") == "diagnostic":
            miss = [k for k in ("version", "source_sha256", "focus") if not it.get(k)]
            if miss or not r:
                print("  %s 定点题缺字段：%s%s"
                      % (it["id"], miss, " 且无 ref 音频" if not r else ""))
                ok = False
        if n == 1:
            print("  %s 单音频题必须改用 ref" % it["id"])
            ok = False
        if n == 0 and not r:
            print("  %s 既无选项音频也无 ref -> 废题" % it["id"])
            ok = False
    if ok:
        print("  题目结构 OK")

    out = os.path.join(FB_DIR, "items.json")
    arc = os.path.join(FB_DIR, "items_roundz.json")
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
