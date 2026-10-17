# -*- coding: utf-8 -*-
"""mk_z23.py —— z23 批次：**变调 vs 不变调** 员工盲听裁决。

## 为什么出这题（z22 之后的实测）

z22 结论：v1.4 与 v2.0 都被员工判 C（明显是机器，0 分），EL 判 B（接近了，1 分）。
L3 全篇实测发现**后处理在损失音色**，主因是变调：

| 链路 | 音色 ×EL | click% | 停顿中位 | F0 | 质心 |
|---|---|---|---|---|---|
| 裸输出 | 0.8995 | 0.0096 | 0.22 | 179.8 | 857 |
| D 变调+温和EQ（v2.0 现役） | 0.7928 | 0.0041 | 0.20 | 187.2 | 789 |
| E 不变调+温和EQ | **0.8835** | **0.0032** | **0.18** | 178.8 | 763 |
| F 不变调+无EQ | **0.9091** | 0.0091 | **0.18** | 176.7 | 831 |
| EL | 1.000 | 0.0124 | 0.16 | 200.7 | 818 |

→ 变调同时伤害**音色（-0.09）/ click / 停顿长度**，是纯亏损；
  不变调拿回音色 12%，但 F0 比 EL 低 ~11%（约 1.8 个半音）。
→ **这是引擎级二选一，不能让工程师替员工决定，必须盲听裁决。**

## 题目

4 段同文案（d172b8c4efb3 男声）同位置 60~120s，单段独立评分。
身份只记在服务端 note，题干/选项不出现「我们的/EL/标杆/v2.0」。

字段名必须与 app/ui/feedback.html 的 render() 对齐：
**title / desc / dim / options[].key**（z22 首版因写成 question/description
且选项缺 key，导致题干不显示、提交 picks 为空、员工全部提交失败）。
"""
import json
import os
import subprocess
import sys
import tempfile

FB = "/mnt/tts/fb"
VER = "v2.0"
ROUND = "z23"
CLIP = 60.0
START = 60.0
TARGET_LUFS = -20.0

FILES = [
    # (输出名, 源, 角色, 中文标签)
    ("z23_m_pitch.wav", "/mnt/tts/diag/l3_52/D_mild_no_clean.wav",
     "pitch", "变调+温和EQ（现役 v2.0）"),
    ("z23_m_nopitch_eq.wav", "/mnt/tts/diag/nopitch52/E_nopitch_mild.wav",
     "nopitch_eq", "不变调+温和EQ"),
    ("z23_m_nopitch_raw.wav", "/mnt/tts/diag/nopitch52/F_nopitch_noeq.wav",
     "nopitch_raw", "不变调+无EQ"),
    ("z23_m_el.wav", "/mnt/tts/data/cases/d172b8c4efb3/merged.wav",
     "el", "EL 标杆"),
]


def cut(src):
    tmp = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-ss", str(START), "-t", str(CLIP),
                    "-ar", "16000", "-ac", "1", tmp], check=True)
    return tmp


def lufs(path):
    """ffmpeg ebur128 集成响度。⚠️ 统计用 info 级别输出，不能传 -v error。"""
    try:
        r = subprocess.run(["ffmpeg", "-v", "info", "-i", path,
                            "-af", "ebur128=framelog=quiet", "-f", "null", "-"],
                           capture_output=True, text=True, timeout=600)
        tail = [l for l in r.stderr.splitlines() if "I:" in l]
        if tail:
            return round(float(tail[-1].split("I:")[1].split()[0]), 1)
    except Exception:
        pass
    return None


def norm_to(src, dst, target=TARGET_LUFS):
    cur = lufs(src)
    if cur is None:
        return None, None
    d = target - cur
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-af", "volume=%gdB" % d, "-ar", "16000", "-ac", "1", dst],
                   check=True)
    return cur, lufs(dst)


Q = "请听这一段，听完后回答：**这段听起来像不像真人在说话？**"
D = ("就当你在手机或车里偶然刷到这条股票广播，随便听，凭第一感觉选。\n"
     "不用分辨细节，也不用和专业主播比。\n"
     "下面三个选项是**评价档位**，不是几段不同的音频——只播一遍，选一个最贴近你感觉的。")

ITEMS = []


def build_items():
    for name, src, role, label in FILES:
        key = name.replace("z23_", "").replace(".wav", "")
        t = cut(src)
        cur, after = norm_to(t, os.path.join(FB, name))
        print("%s  LUFS %.1f -> %.1f" % (name, cur or float("nan"),
                                        after or float("nan")), flush=True)
        ITEMS.append({
            "id": "%s-%s-%s" % (VER, ROUND, key),
            "dim": "男声·整体自然度",
            "title": Q,
            "desc": D,
            "ref": None,
            "note": "性别=male 角色=%s 源=%s 位置=%.0fs 长度=%.0fs" % (
                role, label, START, CLIP),
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
    build_items()
    out = os.path.join(FB, "items_z23.json")
    json.dump({"items": ITEMS}, open(out, "w"), ensure_ascii=False, indent=2)
    print("\n%s  %d 题" % (out, len(ITEMS)), flush=True)

    print("\n自检：")
    ok = True
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
                print("  %s exists=%s size=%d" % (
                    fn, e, os.path.getsize(p) if e else 0), flush=True)
                if not e:
                    ok = False
    print("  %s" % ("✅ 字段/文件自检全过" if ok else "❌ 自检未过，勿切换"))
    print("\n切换命令：cp %s/items_z23.json %s/items.json" % (FB, FB))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
