# -*- coding: utf-8 -*-
"""mk_z25.py —— z25 批次：验证优化点「**换参考音频**」有没有改善。

## 反馈题的唯一目的（用户 2026-10-17 纠正）
> 「你应该反馈的就是，你优化的点，发出来让员工反馈有没有改善，
>  要不然为什么这么多轮的优化的目的是什么」

→ 每轮必须问「**这个优化点有没有改善**」，做法是**同一优化点的前后两版**
  让员工分别评分、比均分。不是测已上线版本，更不是拿 EL 占位。

## 本轮优化点：换参考音频（B 级数据级）

现役参考 `prompt_yohan_1a6c20a6.wav` / `prompt_sola_00b6a45a.wav` 是
gpt-sovits 时代的切片。实测 omni 对参考 F0 有「压缩」：
yohan 参考 F0 196.1 → 输出 176.7（-10%），比 EL 男声 200.7 低 12%。

改用**同案例 EL 分段**（`seg0.wav` 的 5~15s，10s）做参考后，
同链路、同文本、唯一变量只有参考音频，全篇实测：

| 指标 | 男 EL参考 | 男 现役参考 | EL 男 |
|---|---|---|---|
| F0 | 197.2 | 176.7 | 200.7 |
| 音色 ×EL | 0.9671 | 0.9395 | 1.000 |
| 停顿中位 | 0.26s | 0.34s | 0.16s |
| 质心 | 829 | 793 | 818 |
| 1k-3k% | 20.08 | 17.53 | 20.71 |
| click % | 0.0167 | 0.0172 | 0.0124 |

| 指标 | 女 EL参考 | 女 现役参考 | EL 女 |
|---|---|---|---|
| click % | 0.0042 | 0.0073 | 0.0018 |
| 停顿中位 | 0.16s | 0.20s | 0.18s |
| CV | 0.77 | 0.60 | 0.70 |
| 质心 | 722 | 716 | 739 |

→ 男女两侧都是多项改善，值得员工裁决。

🔴 **本轮零 EL**：员工 4 段全是我们自己的版本（新参考 / 现役参考 × 男女）。
   已有 5 轮 EL 基线（z20~z24，1~2 分），拿 EL 占位是浪费员工时间。

## 参考音频硬约束（实测踩坑）
· ref_text **字数必须与音频时长配比**：10s 参考约 85 字（8.5 字/s）。
  首版按句边界向后扩展取到 174 字/10s = 17.4 字/s，**输出只出 46s**（语速崩），
  正是「ref_text 比音频长 → 语速崩」的老坑。
· 只向前找句号对齐起点，**终点按字符数硬切，绝不向后扩展**。
· 生成后必须把 `/voices` 参考**改回原值**，否则污染后续生成。

字段名必须与 app/ui/feedback.html 对齐：title / desc / dim / options[].key。
"""
import json
import os
import subprocess
import sys
import tempfile

FB = "/mnt/tts/fb"
VER = "v3.0"
ROUND = "z25"
CLIP = 60.0
START = 60.0
TARGET_LUFS = -20.0

FILES = [
    # (输出名, 源, 性别, 角色, 中文标签)
    ("z25_m_newref.wav", "/mnt/tts/diag/refchain53/m_elref.wav",
     "male", "new_ref", "男声·新参考"),
    ("z25_m_curref.wav", "/mnt/tts/output/l3-m-omni/l3-m-omni-1.mp3",
     "male", "cur_ref", "男声·现役参考"),
    ("z25_f_newref.wav", "/mnt/tts/diag/refchain53/f_elref.wav",
     "female", "new_ref", "女声·新参考"),
    ("z25_f_curref.wav", "/mnt/tts/output/l3-f-omni-2/l3-f-omni-1.mp3",
     "female", "cur_ref", "女声·现役参考"),
]


def lufs(path):
    """ffmpeg ebur128 集成响度。⚠️ 统计用 info 级别，不能传 -v error。"""
    r = subprocess.run(["ffmpeg", "-v", "info", "-i", path,
                        "-af", "ebur128=framelog=quiet", "-f", "null", "-"],
                       capture_output=True, text=True, timeout=600)
    tail = [l for l in r.stderr.splitlines() if "I:" in l]
    if tail:
        return round(float(tail[-1].split("I:")[1].split()[0]), 1)
    return None


def norm(src, dst, target=TARGET_LUFS):
    """响度对齐。🔴 src 与 dst 相同会炸（ffmpeg 不允许同名输入输出），
    z22/z25 都踩过 —— 内部一律走临时文件。"""
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


def cut(src, dst, start=START, clip=CLIP):
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
    for name, src, gender, role, label in FILES:
        key = name.replace("z25_", "").replace(".wav", "")
        t = os.path.join(FB, name)
        cut(src, t)
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
                gender, role, label, START, CLIP),
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
    out = os.path.join(FB, "items_z25.json")
    json.dump({"items": ITEMS}, open(out, "w"), ensure_ascii=False, indent=2)
    print("\n%s  %d 题（零 EL，全是我们的版本）" % (out, len(ITEMS)), flush=True)
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
                if not e or sz < 16000 * 60 * 2 // 2:
                    print("    ❌ 文件缺失或过小")
                    ok = False
    print("  %s" % ("✅ 字段/文件自检全过" if ok else "❌ 自检未过，勿切换"))
    print("\n切换命令：cp %s/items_z25.json %s/items.json" % (FB, FB))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
