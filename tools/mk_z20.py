# -*- coding: utf-8 -*-
"""mk_z20.py —— z20 批次：**单段独立评分**，用员工的韩语语感，不做「找不同」。

## 为什么要改出题方式（用户 2026-10-17 明确纠正）
> 「员工都是非专业人士都是普通人，只是懂韩语，常年在韩国……要发挥他们的长处，
>   不能做一些找不同的语音，出题角度要友好又能命中要害」

以前的题全是 A/B 对比（「哪一段杂音少」「哪一段音色更像」）——那是**工程师的活**，
让普通人分辨细微差异，既难又不准，还导致控制题大量作废（z18 两人全错、z19 一人错）。

**员工的真正长处是语感**：一段韩语广播听着别扭不别扭、像不像真人主播，
他们常年浸在韩语环境里，这个判断比任何客观指标都准。

→ 改成：**每段独立听，问一个整体印象问题**。不再两两对比，不再有控制题。

## 题目设计
6 段（女声 3 + 男声 3），每段只问一句：

    这段广播听起来像不像真人在说话？
    ① 像真人，很自然
    ② 听得出是 AI，但还算舒服
    ③ 明显是机器，听着别扭

拿到答案怎么用：算**平均分**，三方对比
    我们 vs GS vs EL —— 分差直接回答「换不换引擎」，且是**听感口径**。

## 素材（三方同文案，各切开头 25s，天然对齐）
| 性别 | 现役 | GS | EL |
|---|---|---|---|
| 女 | v11_f_official (omnivoice) | gs48f_eq | cases/00b6a45ad6aa/merged.wav |
| 男 | longprobe-6cdd (voxcpm) | gs48m_post2 | cases/6cdd25bc75f2/merged.wav |

女声同文案「아모텍」3415 字；男声同文案「리노공업」。

🔴 单段评分**必须做响度对齐**：否则员工可能只凭「哪个响」给分，
   评分就变成了比音量。六段统一到 -20 LUFS。

用法：python3 mk_z20.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import soundfile as sf

FB = "/mnt/tts/fb"
VER = "v1.4"
ROUND = "z20"
CLIP = 25.0
TARGET_LUFS = -20.0

FILES = [
    # (输出名, 源, 性别, 身份, 中文标签)
    ("z20_f_cur.wav", "/mnt/tts/output/v11_f_official/v11_f_official-1.mp3",
     "f", "current", "现役 omnivoice"),
    ("z20_f_gs.wav", "/mnt/tts/output/gs48f_eq/gs48f_eq-1.mp3",
     "f", "gs", "GS 女声"),
    ("z20_f_el.wav", "/mnt/tts/data/cases/00b6a45ad6aa/merged.wav",
     "f", "el", "EL 女声"),
    ("z20_m_cur.wav", "/mnt/tts/output/longprobe-6cdd/longprobe-6cdd-1.mp3",
     "m", "current", "现役 voxcpm"),
    ("z20_m_gs.wav", "/mnt/tts/output/gs48m_post2/gs48m_post2-1.mp3",
     "m", "gs", "GS 男声"),
    ("z20_m_el.wav", "/mnt/tts/data/cases/6cdd25bc75f2/merged.wav",
     "m", "el", "EL 男声"),
]


def cut(src, out, clip=CLIP):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-t", str(clip), "-ar", "16000", "-ac", "1", t], check=True)
    return t


def lufs(path):
    r = subprocess.run(["ffmpeg", "-v", "info", "-i", path,
                        "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    v = None
    for line in r.splitlines():
        if "I:" in line and "LUFS" in line:
            try:
                v = float(line.split("I:")[1].split("LUFS")[0].strip())
            except Exception:
                pass
    return v


def norm_to(src, dst, target=TARGET_LUFS):
    """两遍 loudnorm：先测再改，保证六段响度一致。"""
    cur = lufs(src)
    if cur is None:
        shutil.copyfile(src, dst)
        return None, None
    d = target - cur
    # 🔴 单位是 `dB` 不是 `dBB`！写成 %gdBB 时 ffmpeg 解析不出正确单位，
    #    增益会离谱（实测 -23.1 -> -5.7 LUFS，六段全被拉爆，原始差异也没消掉）。
    #    这个坑在规范「自检链」里记过一次，照抄时又踩了。
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-af", "volume=%gdB" % d, "-ar", "16000", "-ac", "1", dst],
                   check=True)
    return cur, lufs(dst)


Q = "请听这一段，听完后回答：**这段听起来像不像真人在说话？**"
D = ("就当你在刷手机时偶然刷到这条股票广播，随便听，凭第一感觉选。\n"
     "不用分辨细节，也不用和专业主播比。\n"
     "（这一批里既有女声也有男声，一段一段听就行。）\n"
     "下面三个选项是**评价档位**，不是三段不同的音频——只播一遍，选一个最贴近你感觉的。")

ITEMS = []
for name, src, gender, role, label in FILES:
    key = name.replace("z20_", "").replace(".wav", "")
    ITEMS.append({
        "id": "v1.4-%s-%s" % (ROUND, key),
        "dim": "%s声·整体自然度" % ("女" if gender == "f" else "男"),
        "title": Q,
        "desc": D,
        "ref": None,
        "note": "%s声 %s（%s）。\n源：%s，开头 %.0fs。\n"
                "🔴 出题方式（2026-10-17 用户纠正）：员工是普通人、只懂韩语，"
                "不做 A/B「找不同」，只问整体印象，用他们的语感。\n"
                "六段统一响度到 %.1f LUFS，避免员工只凭音量给分。\n"
                "评分：①=2 分 ②=1 分 ③=0 分，按性别分组算平均分，"
                "三方（现役 / GS / EL）对比直接回答换不换引擎。"
                % ("女" if gender == "f" else "男", label, role,
                   os.path.basename(src), CLIP, TARGET_LUFS),
        "ver": VER,
        # 🔴 只有 A 带 file：前端对每个有 file 的 option 都渲染一个播放器，
        #    三个都带会播三遍同一段。B/C 只作评价档位。
        "options": [
            {"key": "A", "file": name, "label": "① 像真人，很自然"},
            {"key": "B", "file": None, "text": "（同上，只是评价档位）",
             "label": "② 听得出是 AI，但还算舒服"},
            {"key": "C", "file": None, "text": "（同上，只是评价档位）",
             "label": "③ 明显是机器，听着别扭"},
        ],
    })


def main():
    print("口径：单段独立评分，无 A/B 对比，无控制题；响度统一 %.1f LUFS" % TARGET_LUFS)
    for name, src, gender, role, label in FILES:
        if not os.path.exists(src):
            print("❌ 缺源", src)
            return 1
        t = cut(src, name)
        dst = os.path.join(FB, name)
        a, b = norm_to(t, dst)
        dur = sf.info(dst).duration
        print("  %-16s %-14s 时长 %.1fs  LUFS %.2f -> %.2f"
              % (name, label, dur, a if a else 0, b if b else 0))

    # 自检：文件齐全 + 时长一致 + md5 不撞车
    import hashlib
    hs = {}
    for name, *_ in FILES:
        p = os.path.join(FB, name)
        h = hashlib.md5(open(p, "rb").read()).hexdigest()[:8]
        hs.setdefault(h, []).append(name)
    dup = {k: v for k, v in hs.items() if len(v) > 1}
    if dup:
        print("❌ md5 撞车:", dup)
        return 1
    durs = [sf.info(os.path.join(FB, n)).duration for n, *_ in FILES]
    if max(durs) - min(durs) > 0.5:
        print("❌ 时长不一致:", durs)
        return 1
    print("✅ 自检通过：6 段，md5 不撞车，时长 %.1f±%.1fs"
          % (sum(durs) / len(durs), max(durs) - min(durs)))

    # 备份旧批次（用户要求 z19 全部作废）
    ts = time.strftime("%Y%m%d_%H%M%S")
    for f in ("items.json", "votes.jsonl"):
        p = os.path.join(FB, f)
        if os.path.exists(p):
            shutil.copyfile(p, p + ".bak-" + ts)
            print("备份", f, "->", f + ".bak-" + ts)

    out = os.path.join(FB, "items_%s.json" % ROUND)
    with open(out, "w", encoding="utf-8") as fp:
        json.dump({"items": ITEMS}, fp, ensure_ascii=False, indent=1)
    shutil.copyfile(out, os.path.join(FB, "items.json"))
    print("写入", out, "并上线，共", len(ITEMS), "题")
    return 0


if __name__ == "__main__":
    sys.exit(main())
