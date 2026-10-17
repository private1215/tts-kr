# -*- coding: utf-8 -*-
"""mk_z21.py —— z21 批次：**定位题**，问「最让你觉得不像真人的是哪一点」。

## 为什么出这批题

z20（2 人独立投票）结果：
    女声 现役 1.50 | GS 0.00 | EL 2.00
    男声 现役 0.50 | GS 0.00 | EL 2.00

结论有两条，都指向「必须再问一轮」：

1. **换 GS 的路堵死了**：GS 客观指标全面优于 EL（click 0.6×EL、音色 0.92、
   杂音 4.12、CER 0.053），员工却一致判「明显是机器」。
   → click / 音色 / 杂音**不能代表听感**，不能拿它们决定换引擎。
2. **主战场是男声**：现役 0.5 vs EL 2.0，差 1.5 分（女声只差 0.5）。

但我们**不知道男声到底哪里别扭**。客观最大短板是 click（4.24 次/秒 = EL 的
9.6 倍），可「客观对得上听感」这件事刚刚才被 GS 打过脸，不能直接信。

→ 直接问韩语母语者。他们的长处是语感，让他们指认现象，我们再去改对应的东西。

## 出题原则（第50轮用户明确纠正过）

- 不做「找不同」：不给两段音频让员工比较。
- 单段独立听，问**印象/感受**，不是问技术细节。
- 盲听：选项里不出现「我们的 / EL / 标杆 / 参考」字样。
- 每个选项都必须**对应一个可执行的下一步**（否则不出这道题）。

## 选项 → 拿到答案后改什么

| 选项 | 现象 | 下一步 |
|---|---|---|
| A | 声音本身像机器音 | 换引擎 / 再训练（音色层面） |
| B | 句子断开处有异声 | 修断句处异声（⚠️ 已多次尝试失败，先确认是否真听得见） |
| C | 有杂音、爆音 | 修 click —— 目前**唯一**客观与听感同向的指标 |
| D | 语速或节奏别扭 | 调韵律参数（语速/停顿） |

## 素材

现役男女声各一段，**取 60~95s**（避开 z20 用过的开头 25s，免得员工觉得重复）。
统一 35s / 16k / -20 LUFS（与 z20 同一口径）。

用法：python3 mk_z21.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import soundfile as sf

FB = "/mnt/tts/fb"
VER = "v1.4"
ROUND = "z21"
START = 60.0        # 避开 z20 用过的开头 25s
CLIP = 35.0         # 比 z20 的 25s 长一点，定位需要多听几句
TARGET_LUFS = -20.0

FILES = [
    # (输出名, 源, 性别, 身份, 中文标签)
    ("z21_male.wav",
     "/mnt/tts/output/longprobe-6cdd/longprobe-6cdd-1.mp3",
     "m", "current", "男声现役 voxcpm"),
    ("z21_female.wav",
     "/mnt/tts/output/v11_f_official/v11_f_official-1.mp3",
     "f", "current", "女声现役 omnivoice"),
]


def cut(src, out, start=START, clip=CLIP):
    t = tempfile.mktemp(suffix=".wav")
    # -ss 放在 -i 前 = 输入快 seek，比放在后面快很多
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(start),
                    "-i", src, "-t", str(clip), "-ar", "16000", "-ac", "1", t],
                   check=True)
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
    cur = lufs(src)
    if cur is None:
        shutil.copyfile(src, dst)
        return None, None
    d = target - cur
    # 🔴 单位是 dB 不是 dBB（写错会把音频拉到 -5.7 LUFS 爆音，这个坑踩过两次）
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-af", "volume=%gdB" % d, "-ar", "16000", "-ac", "1", dst],
                   check=True)
    return cur, lufs(dst)


Q = "请再听这一段，回答：**最让你觉得「不像真人」的是哪一点？**"
D = ("还是随便听，凭第一感觉。\n"
     "不用分析技术细节，也不用和专业主播比。\n"
     "只选**最突出**的那一处——如果好几处都有，就选你最先注意到的那个。\n"
     "下面四个选项是**四种不同的问题**，不是四段音频——只播一遍，选一个。\n"
     "如果有别的感受，请在最后留言里写出来，对我们很有帮助。")

# 🔴 每个选项都对应一个可执行的下一步，见文件头表格
OPTS = [
    ("A", "声音本身像机器音（音色、音质不像真人说话）"),
    ("B", "句子断开的地方有异声（断句处冒出奇怪的短音）"),
    ("C", "有杂音、爆音（咔哒、滋啦之类的声音）"),
    ("D", "说话的速度或节奏别扭（快慢、停顿不自然）"),
]

ITEMS = []
for name, src, gender, role, label in FILES:
    key = name.replace("z21_", "").replace(".wav", "")
    ITEMS.append({
        "id": "v1.4-%s-%s" % (ROUND, key),
        "dim": "%s声·问题定位" % ("男" if gender == "m" else "女"),
        "title": Q,
        "desc": D,
        "ref": None,
        "note": "%s（%s）。\n源：%s，%.0f~%.0fs。\n"
                "🔴 定位题（z20 之后）：GS 已出局，主战场是男声（0.5 vs EL 2.0）。\n"
                "选项 → 下一步：A=换引擎/再训练；B=修断句异声（已多次失败，先验证可闻性）；\n"
                "C=修 click（唯一客观与听感同向的指标）；D=调韵律参数。\n"
                "出题方式（2026-10-17 用户纠正）：不做「找不同」，单段独立听，只问感受。\n"
                "响度统一 %.1f LUFS，避免员工只凭音量判断。"
                % (label, role, os.path.basename(src), START, START + CLIP,
                   TARGET_LUFS),
        "ver": VER,
        # 🔴 只有第一个选项带 file：前端对每个有 file 的 option 都渲染一个播放器，
        #    四个都带会播四遍同一段。其余只作文字选项。
        "options": [
            {"key": k,
             "file": name if i == 0 else None,
             "text": None if i == 0 else "（同上一段音频，只是不同的问题选项）",
             "label": lab}
            for i, (k, lab) in enumerate(OPTS)
        ],
    })


def main():
    print("口径：定位题（单选最突出问题），现役男女声各一段，"
          "%.0f~%.0fs，响度 %.1f LUFS" % (START, START + CLIP, TARGET_LUFS))
    made = []
    for name, src, gender, role, label in FILES:
        if not os.path.exists(src):
            print("❌ 缺源", src)
            return 1
        t = cut(src, name)
        dst = os.path.join(FB, name)
        a, b = norm_to(t, dst)
        made.append(dst)
        print("  %-16s %-18s 时长 %.1fs  LUFS %.2f -> %.2f"
              % (name, label, sf.info(dst).duration, a or 0, b or 0))

    # ---- 自检 ----
    import hashlib
    md5s = {}
    ok = True
    for p in made:
        md5s[p] = hashlib.md5(open(p, "rb").read()).hexdigest()
    if len(set(md5s.values())) != len(md5s):
        print("❌ md5 撞车")
        ok = False
    for p in made:
        d = sf.info(p).duration
        if abs(d - CLIP) > 1.0:
            print("❌ 时长异常 %s %.1fs" % (p, d))
            ok = False
        if abs((lufs(p) or -99) - TARGET_LUFS) > 0.5:
            print("❌ 响度异常 %s" % p)
            ok = False

    out = os.path.join(FB, "items_z21.json")
    json.dump({"items": ITEMS}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n题库 -> %s（%d 题）" % (out, len(ITEMS)))
    print("自检：%s" % ("✅ 通过" if ok else "❌ 失败"))
    print("\n上线：cp items_z21.json items.json（先备份 items.json / votes.jsonl）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
