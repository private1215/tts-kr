# -*- coding: utf-8 -*-
"""mk_z29.py —— z29 批次：验证「句末粘连型嗯清理」是否解决男声卡顿/断句异声。

## 上一轮（z28）员工反馈给了关键定位

员工 A 给了具体位置：
    男 v4（80s 起点）：5s / 21s / 49s / 1分03s -> 绝对 85 / 101 / 129 / 143
    女 v3（340s 起点）：10s / 23s / 34s / 47~50s / 1分 / 1分08s
                     -> 绝对 350 / 363 / 374 / 387~390 / 400 / 408

## 定位：「嗯」分两类

**孤立型**（v1~v4 一直在清）：夹在静音之间，gruntclean53.py 已清。
**句末粘连型**（用户 18:33 就指出，我不该忘掉的那句）：
    说完一个字后紧跟的浊音尾巴，如：
    男 101s -> 100.66s 30ms(-33dB, zcr=0.027, 低频=0.90)
    男 143s -> 143.28~143.36s 10+60+5ms 浊音残片
    女 374s -> 373.40s 80ms(-33.6dB, zcr=0.042, 低频=0.70)

`tailgrunt53e.py` 全篇统计：
    男：我们 49 个 (5.25/分钟) vs EL 36 个 (3.89/分钟) → 我们多 36%
    女：我们 14 个 (1.83/分钟) vs EL 34 个 (4.44/分钟) → 我们比 EL 少

→ 男声「嗯/卡顿/断句异声」的根因是句末粘连型浊音尾巴。
→ 女声方向反，不动；员工说的「女声有回音」是另一类问题，本轮不处理。

## 修复：tailclean53.py

用 -40dB 绝对门槛分段，检测：
    · 时长 5~120ms
    · 峰值 -45 ~ -20dB
    · 浊音：zcr < 0.15 且 300Hz 以下能量 > 0.45
    · 紧连前一段（间隔 < 0.03s）
    · 后段开始前有 >0.05s 间隙
    · 前一段时长 > 0.10s

用等长静音替换（总时长不变），且直接填 0（避免残留电平再被检测）。
男声清理 49 处；再叠加 gruntclean v4 又清 17 处孤立型。

## 全篇实测（男声）

| 指标 | 原始 | v4 | v4+tail |
|---|---|---|---|
| click % | 0.0172 | 0.0172 | 0.0172 |
| 停顿中位 | 0.34s | 0.34s | **0.32s** |
| 停顿数 | 369 | 358 | 367 |
| 质心 | 793 | 781 | 782 |
| cps | 8.20 | 8.20 | 8.20 |

停顿数从 358 回到 367（把 49 个粘连尾巴变成静音，等于增加了 49 处停顿）。

## 题目（零 EL）

4 段：男 v4 / 男 v4+tail / 女 v4 / 女 v3。
男声片段仍是 80~170s（含多个 tail 嗯），直接和 z28 男 v4 比分数。
女声保留 v4/v3 作为参考，看是否有变化。
"""
import json
import os
import subprocess
import sys
import tempfile

FB = "/mnt/tts/fb"
VER = "v3.0"
ROUND = "z29"
TARGET_LUFS = -20.0

FILES = [
    ("z29_m_v4.wav", "/mnt/tts/diag/gruntclean53/v4/m_clean.wav",
     "male", "v4", "男声·v4整岛抹平(18处)", 80.0, 90.0),
    ("z29_m_tail.wav", "/mnt/tts/diag/tailclean53/m_tail2_grunt.wav",
     "male", "tail", "男声·v4+tail(49+17处)", 80.0, 90.0),
    ("z29_f_v4.wav", "/mnt/tts/diag/gruntclean53/v4/f_clean.wav",
     "female", "v4", "女声·v4整岛抹平(5处)", 340.0, 70.0),
    ("z29_f_v3.wav", "/mnt/tts/diag/gruntclean53/v3/f_clean.wav",
     "female", "v3", "女声·v3扩展12dB(5处)", 340.0, 70.0),
]


def lufs(path):
    r = subprocess.run(["ffmpeg", "-v", "info", "-i", path,
                        "-af", "ebur128=framelog=quiet", "-f", "null", "-"],
                       capture_output=True, text=True, timeout=600)
    tail = [l for l in r.stderr.splitlines() if "I:" in l]
    if tail:
        return round(float(tail[-1].split("I:")[1].split()[0]), 1)
    return None


def norm(src, dst, target=TARGET_LUFS):
    cur = lufs(src)
    if cur is None:
        return None, None
    d = target - cur
    tmp = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-af", "volume=%gdB" % d, "-ar", "16000", "-ac", "1", tmp],
                   check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp,
                    "-ar", "16000", "-ac", "1", dst], check=True)
    try:
        os.unlink(tmp)
    except OSError:
        pass
    return cur, lufs(dst)


def cut(src, dst, start, clip):
    tmp = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-ss", str(start), "-t", str(clip),
                    "-ar", "16000", "-ac", "1", tmp], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp,
                    "-ar", "16000", "-ac", "1", dst], check=True)
    return dst


Q = "请听这一段，听完后回答：**这段听起来像不像真人在说话？**"
D = ("就当你在手机或车里偶然刷到这条股票广播，随便听，凭第一感觉选。\n"
     "不用分辨细节，也不用和专业主播比。\n"
     "下面三个选项是**评价档位**，不是几段不同的音频——只播一遍，选一个最贴近你感觉的。")

ITEMS = []


def build():
    for name, src, gender, role, label, start, clip in FILES:
        key = name.replace("z29_", "").replace(".wav", "")
        t = os.path.join(FB, name)
        cut(src, t, start, clip)
        cur, after = norm(t, t)
        print("%s  LUFS %.1f -> %.1f" % (name, cur or float("nan"),
                                        after or float("nan")), flush=True)
        ITEMS.append({
            "id": "%s-%s-%s" % (VER, ROUND, key),
            "dim": "%s声·整体自然度" % ("男" if gender == "male" else "女"),
            "title": Q,
            "desc": D,
            "ref": None,
            "note": "性别=%s 角色=%s 源=%s 位置=%.0fs 长度=%.0fs" % (
                gender, role, label, start, clip),
            "ver": VER,
            "options": [
                {"key": "A", "label": "① 像真人，很自然", "file": name},
                {"key": "B", "label": "② 听得出是 AI，但还算舒服",
                 "file": None, "text": "（同上，只是评价档位）"},
                {"key": "C", "label": "③ 明显是机器，听着别扭",
                 "file": None, "text": "（同上，只是评价档位）"},
            ]
        })


def main():
    os.makedirs(FB, exist_ok=True)
    build()
    out = os.path.join(FB, "items_z29.json")
    json.dump({"items": ITEMS}, open(out, "w"), ensure_ascii=False, indent=2)
    print("\n%s  %d 题（零 EL）" % (out, len(ITEMS)), flush=True)
    for i, it in enumerate(ITEMS, 1):
        print("  %d. %-22s %s" % (i, it["id"],
                                  it["note"].split("源=")[-1]), flush=True)

    ok = True
    print("\n自检：")
    for it in ITEMS:
        for k in ("id", "title", "desc", "dim", "options"):
            if not it.get(k):
                print("  ❌ 缺字段 %s: %s" % (k, it.get("id")))
                ok = False
        for o in it["options"]:
            if not o.get("key"):
                print("  ❌ 选项缺 key: %s" % o.get("label"))
                ok = False
            fn = o.get("file")
            if fn:
                p = os.path.join(FB, fn)
                e = os.path.exists(p)
                sz = os.path.getsize(p) if e else 0
                print("  %s exists=%s size=%d" % (fn, e, sz), flush=True)
                if not e or sz < 16000 * 30 * 2:
                    print("    ❌ 文件缺失或过小")
                    ok = False
    print("  %s" % ("✅ 字段/文件自检全过" if ok else "❌ 自检未过，勿切换"))
    print("\n切换命令：cp %s/items_z29.json %s/items.json" % (FB, FB))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
