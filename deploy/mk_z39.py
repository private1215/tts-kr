#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第三十九轮题库：**女声专场**（首次给女声打基线 + 高频衰减 EQ 裁决）

## 为什么开女声战线
男声（yohan）这条线已推进到「块长 300 + h5」，员工仍说有电子音，
剩余缺口在引擎（音色均值 0.781 vs EL 0.934）→ 交 GS 微调（正在后台训）。
**女声（SOLA）一直没测过**，而客观数据显示女声的短板方向和男声相反：

| 指标 | 女声现役 | EL 女声 |
|---|---|---|
| 平坦度 | **+3.85σ**（偏噪） | 0 |
| 4k~8k | **+2.13σ**（高频偏多） | 0 |
| Σz² | 32.2 | — |
| 音色均值 | 0.689 | — |

🔴 男声是「偏干净、不能压高频」，女声是「偏噪、正该压高频」——方向相反，
   男声那条教训（窄带 EQ 打塌平坦度）**不适用**于女声。

## EQ 细扫（198s 女声成品，纯后处理，不占 GPU）
| 衰减 | Σz² | 4k~8k | 平坦度 |
|---|---|---|---|
| 0（现役） | 32.2 | +2.13σ | +3.85σ |
| −1.5dB | 19.1 | −0.64σ | +3.06σ |
| **−2dB** | **18.9** | **−1.45σ** | +2.81σ |
| −2.5dB | 20.0 | −2.21σ | +2.55σ |
| −3dB | 22.2 | −2.92σ | +2.30σ |
| −4dB | 29.6 | −4.21σ | +1.78σ |

→ 取 **−2dB**：Σz² −41%，4k~8k 基本对齐 EL，**音色完全不变**（0.6886 vs 0.6887）。
   EQ 是后处理且不吃音色，所以不用重新生成，直接切片段给员工听即可。

## 题（4 题）
1. 女声基线（我们 vs EL，开头 30s）——首次建立女声锚点
2. 女声裁决（现役 vs 加 EQ，中段 30s）——决定 EQ 是否上线
3. 女声裁决（现役 vs 加 EQ，尾部 20s）——尾部是男声的短板，女声也验
4. 控制题
"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
TMPD = "/mnt/tts/diag/_z39"
TGT_LUFS = -26.1
SR_CLIP = 24000

OURS = "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"   # 女声现役成品 198.7s
EL_S = "/mnt/tts/data/cases/52727ec43154/merged.wav"
EQ_F = "highshelf=f=4000:g=-2"          # 本轮候选

SUFFIX = "20261016z7"

CLIPS = [
    ("z39_head_a", OURS, 0.0, 30.0, ""),        # 我们 开头
    ("z39_head_b", EL_S, 0.0, 30.0, ""),        # EL 开头
    ("z39_mid_a", OURS, 90.0, 30.0, ""),        # 我们 中段（现役）
    ("z39_mid_b", OURS, 90.0, 30.0, EQ_F),      # 我们 中段（加 EQ）
    ("z39_tail_a", OURS, 165.0, 20.0, ""),      # 我们 尾部（现役）
    ("z39_tail_b", OURS, 165.0, 20.0, EQ_F),    # 我们 尾部（加 EQ）
]


def lufs(p):
    r = subprocess.run(["ffmpeg", "-i", p, "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    v = [l for l in r.stderr.splitlines() if "I:" in l and "LUFS" in l]
    return float(v[-1].split("I:")[1].split("LUFS")[0].strip()) if v else None


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()[:8]


def clip(src, s0, dur, af=""):
    t = os.path.join(TMPD, "_c%s.wav" % (hashlib.md5(
        ("%s%s%s" % (src, s0, af)).encode()).hexdigest()[:6]))
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", src, "-ss", "%.1f" % s0,
           "-ar", str(SR_CLIP), "-ac", "1", "-t", "%.1f" % dur]
    if af:
        cmd += ["-af", af]
    cmd += [t]
    subprocess.run(cmd, check=True)
    got = os.path.getsize(t) / 2.0 / SR_CLIP
    if got < dur - 0.5:
        sys.exit("切片越界：%s 第%.0fs 起取%.0fs，实际仅%.1fs"
                 % (os.path.basename(src), s0, dur, got))
    return t


def emit(wav, name, title):
    g = TGT_LUFS - lufs(wav)
    out = os.path.join(FB_DIR, "%s.mp3" % name)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav,
                    "-af", "volume=%gdB" % g, "-b:a", "192k",
                    "-metadata", "title=%s" % title, out], check=True)
    back = lufs(out)
    print("  %-14s md5=%s  LUFS %6.1f %s"
          % (name, md5(out), back if back else 0,
             "OK" if back and abs(back - TGT_LUFS) < 1.0 else "!!爆表"))
    return out


def like_item(iid, title, desc, a, b, note):
    return {
        "id": iid, "dim": "女声", "title": title, "desc": desc,
        "ref": None, "note": note,
        "options": [
            {"key": "A", "file": a, "label": "① 第 1 段更像真人"},
            {"key": "B", "file": b, "label": "② 第 2 段更像真人"},
            {"key": "C", "file": None, "label": "③ 两段差不多，听不出区别"},
            {"key": "D", "file": None, "label": "④ 两段都像机器，都不像真人"}]}


def main():
    for p in (OURS, EL_S):
        if not os.path.exists(p):
            print("缺素材：%s" % p)
            return 1
    os.makedirs(TMPD, exist_ok=True)
    made = {}
    print("=== 切片 + 归一（女声）===")
    for nm, src, s0, dur, af in CLIPS:
        made[nm] = emit(clip(src, s0, dur, af), nm, nm)
    made["z39_ctrl1"] = emit(clip(OURS, 60.0, 10.0), "z39_ctrl1", "ctrl A")
    made["z39_ctrl2"] = emit(clip(OURS, 60.0, 10.0), "z39_ctrl2", "ctrl B")

    print("\n=== md5 自检 ===")
    inv = {}
    for nm, p in made.items():
        inv.setdefault(md5(p), []).append(nm)
    bad = False
    for h, ns in inv.items():
        if len(ns) > 1:
            print("  %-9s 撞车 -> %s" % (h, ns))
            bad = True
    if not bad:
        print("  无撞车")

    items = [
        {"id": "z7_send_baseline_f_%s" % SUFFIX, "dim": "女声发客户基线（首次）",
         "title": "【女声】这 30 秒，哪一段可以直接发给客户？",
         "desc": "两段音频各 30 秒，都是**女声**，听完选一个。\n\n"
                 "① 第 1 段可以\n② 第 2 段可以\n③ 都不行",
         "ref": None,
         "note": "🔴 女声首次锚点，且**首次改为盲听**：选项不写「我们的/EL」"
                 "（用户要求：标了身份就会主观臆断）。后台口径 A=我们的女声开头 30s，"
                 "B=EL 女声开头 30s。\n"
                 "⚠️ 男声基线那 22 票是在**标了身份**的口径下拿的，含金量存疑；"
                 "盲听口径从这里重新开始记，男声也应在下一轮重测。",
         "options": [
             {"key": "A", "file": "z39_head_a.mp3", "label": "① 第 1 段"},
             {"key": "B", "file": "z39_head_b.mp3", "label": "② 第 2 段"},
             {"key": "C", "file": None, "label": "③ 都不行"}]},
        like_item("z7_feq_mid_%s" % SUFFIX,
                  "【女声·中段】这两段（各 30 秒），哪一段更像「真人在自然说话」？",
                  "两段是**同一版女声**，第 2 段做了一点音质调整。\n"
                  "凭第一感觉选，重点听：有没有沙沙的噪音感、尖不尖。",
                  "z39_mid_a.mp3", "z39_mid_b.mp3",
                  "裁决：女声加不加「高频 −2dB」（现役① vs 加EQ②）。\n"
                  "选② → 上线 eq_female_tmpl 加 highshelf=4000:-2（Σz² 32.2→18.9、"
                  "4k~8k +2.13σ→−1.45σ、平坦度 +3.85σ→+2.81σ、音色不变）。\n"
                  "选①/③ → 不上线，女声另找短板。"),
        like_item("z7_feq_tail_%s" % SUFFIX,
                  "【女声·尾部】这两段（各 20 秒），哪一段更像「真人在自然说话」？",
                  "两段是**同一版女声**的结尾部分，第 2 段做了同样的音质调整。\n"
                  "（男声的短板就在尾部，所以女声也单独验一下。）",
                  "z39_tail_a.mp3", "z39_tail_b.mp3",
                  "女声尾部同款裁决：确认 EQ 在尾部也成立（不是只在开头有效）。\n"
                  "若中段选②而尾部选① → EQ 只能上中段，需分窗处理。"),
        {"id": "z7_ctrl_same_%s" % SUFFIX, "dim": "答题有效性校验",
         "title": "【最后一题】这两段，听起来有区别吗？",
         "desc": "**这道题必须选一个**（选错也没关系）。\n\n"
                 "这两段是**同一段音频放了两遍**，内容完全一样。听 5 秒就够。",
         "ref": None,
         "note": "隐藏控制题：正解=C 两段一样。",
         "options": [
             {"key": "A", "file": "z39_ctrl1.mp3", "label": "A 明显有区别"},
             {"key": "B", "file": "z39_ctrl2.mp3", "label": "B 明显有区别"},
             {"key": "C", "file": None, "label": "C 没有区别，是一样的"}]},
    ]

    print("\n=== 题目自检 ===")
    ok = True
    for it in items:
        n = sum(1 for o in it["options"] if o.get("file"))
        r = (it.get("ref") or {}).get("file")
        for o in it["options"]:
            if o.get("file") and not os.path.exists(os.path.join(FB_DIR, o["file"])):
                print("  %s 缺文件 %s" % (it["id"], o["file"]))
                ok = False
        if n == 0 and not r:
            print("  %s 既无选项音频也无 ref -> 废题" % it["id"])
            ok = False
        if n == 1:
            print("  %s 单音频题必须改用 ref" % it["id"])
            ok = False
        # 🔴 盲听铁律：不得让员工看出哪段是我们的、哪段是 EL
        if n >= 2:
            blob = it["title"] + it["desc"] + "".join(
                o["label"] for o in it["options"])
            for kw in ("我们的", "EL", "标杆", "参考", "标准"):
                if kw in blob:
                    print("  %s 泄漏身份（盲听题）：%s" % (it["id"], kw))
                    ok = False
    if ok:
        print("  题目结构 OK（女声基线1 + 女声裁决2 + 控制1）")

    out = os.path.join(FB_DIR, "items.json")
    arc = os.path.join(FB_DIR, "items_roundz6.json")
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
