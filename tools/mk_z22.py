# -*- coding: utf-8 -*-
"""mk_z22.py —— z22 批次：验证 v2.0 男声 M3+M4 的可闻性。

## 设计

延续 z20 的**单段独立评分**（普通人友好，不做「找不同」）。

三段同文案（d172b8c4efb3，男声「리노공업」），同位置 60~120s，
分别来自：
- v1.4 旧链路（激进 EQ + 停顿清洗）
- v2.0 新链路（温和 EQ + 半极简，跳过 clean_gaps/roomtone）
- EL 标杆

每段只问：**这段听起来像不像真人在说话？**
① 像真人，很自然  ② 听得出是 AI，但还算舒服  ③ 明显是机器，听着别扭

评分映射：①=2 分 / ②=1 分 / ③=0 分。

## 盲听铁律

题干/选项**不出现「我们的 / EL / 标杆 / v1.4 / v2.0」**，身份只在服务端 note 记。
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
VER = "v2.0"
ROUND = "z22"
# L2 长度（60~120s），方向筛查 + 可闻性验证
CLIP = 60.0
# 起点避开开头 60s，与 z20 不重叠
START = 60.0
TARGET_LUFS = -20.0

FILES = [
    # (输出名, 源, 角色, 中文标签)
    ("z22_m_v14.wav", "/mnt/tts/diag/l3_52/A_baseline.wav",
     "v14", "旧链路 v1.4"),
    ("z22_m_v20.wav", "/mnt/tts/diag/l3_52/D_mild_no_clean.wav",
     "v20", "新链路 v2.0"),
    ("z22_m_el.wav", "/mnt/tts/data/cases/d172b8c4efb3/merged.wav",
     "el", "EL 标杆"),
]


def cut(src, out, start=START, clip=CLIP):
    """切指定区间并统一降到 16k 单声道。"""
    tmp = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                  "-ss", str(start), "-t", str(clip),
                  "-ar", "16000", "-ac", "1", tmp], check=True)
    return tmp


def lufs(path):
    """ffmpeg ebur128 集成响度。兼容两种输出格式。
    ⚠️ ebur128 统计用 info 级别输出，不能传 -v error。"""
    try:
        r = subprocess.run(["ffmpeg", "-v", "info", "-i", path,
                            "-af", "ebur128=framelog=quiet", "-f", "null", "-"],
                           capture_output=True, text=True, timeout=600)
        tail = [l for l in r.stderr.splitlines() if "I:" in l]
        if tail:
            return round(float(tail[-1].split("I:")[1].split()[0]), 1)
        for l in r.stderr.splitlines():
            if "Integrated loudness" in l:
                return round(float(l.split("I:")[-1].replace("LUFS", "").strip()), 1)
    except Exception:
        pass
    return None


def norm_to(src, dst, target=TARGET_LUFS):
    """两遍 loudnorm，保证三段响度一致。"""
    cur = lufs(src)
    if cur is None:
        shutil.copyfile(src, dst)
        return None, None
    d = target - cur
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-af", "volume=%gdB" % d, "-ar", "16000", "-ac", "1", dst],
                   check=True)
    return cur, lufs(dst)


Q = "请听这一段，听完后回答：**这段听起来像不像真人在说话？**"
D = ("就当你在手机或车里偶然刷到这条股票广播，随便听，凭第一感觉选。\n"
     "不用分辨细节，也不用和专业主播比。\n"
     "下面三个选项是**评价档位**，不是三段不同的音频——只播一遍，选一个最贴近你感觉的。")

# 🔴🔴 字段名必须与 app/ui/feedback.html 的 render() 对齐：
#     title / desc / dim / options[].key（缺一不可）
#   · 前端 `it.title`、`it.desc` 空 → 题干不显示，只剩选项
#   · 前端 `o.key` 空 → `PICK[id] = ''` 是 falsy → 计数 0/N、
#     提交时 picks 被 `if (PICK[it.id])` 过滤成空对象 → 「还没选，也没写反馈」
#   z22 首版踩过这个坑（用户截图发现），字段名照抄 mk_z20.py。
ITEMS = []


def build_items():
    """切音频 + 归一响度 + 组装题目。字段名必须与 feedback.html 对齐。"""
    for name, src, role, label in FILES:
        key = name.replace("z22_", "").replace(".wav", "")
        t = cut(src, os.path.join(FB, name))
        cur, after = norm_to(t, os.path.join(FB, name))
        print("%s  LUFS %.1f -> %.1f" % (name, cur or float("nan"),
                                        after or float("nan")), flush=True)
        # 🔴🔴 前端单选：选中先清所有 .sel；每个有 file 的 option 都渲染一个播放器。
        #    所以**只让第一个选项带 file**，其余纯文字档位。
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
    out = os.path.join(FB, "items_z22.json")
    json.dump({"items": ITEMS}, open(out, "w"),
              ensure_ascii=False, indent=2)
    print("\n%s  %d 题" % (out, len(ITEMS)), flush=True)

    # 自检：字段齐全 + 文件存在
    print("\n自检：")
    for it in ITEMS:
        for k in ("id", "title", "desc", "dim", "options"):
            if not it.get(k):
                print("  ❌ 缺字段 %s: %s" % (k, it.get("id")))
        for o in it["options"]:
            if not o.get("key"):
                print("  ❌ 选项缺 key: %s" % o.get("label"))
            fn = o.get("file")
            if fn:
                p = os.path.join(FB, fn)
                print("  %s exists=%s size=%d" % (
                    fn, os.path.exists(p),
                    os.path.getsize(p) if os.path.exists(p) else 0), flush=True)
    print("\n切换命令：cp %s/items_z22.json %s/items.json" % (FB, FB))
    return 0


if __name__ == "__main__":
    sys.exit(main())
