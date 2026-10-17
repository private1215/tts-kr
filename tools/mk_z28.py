# -*- coding: utf-8 -*-
"""mk_z28.py —— z28 批次：验证 v4（整岛抹平）是否把「嗯」彻底清干净。

## 演进链（每轮员工都说「清了一部分没清干净」）

| 版本 | 做法 | 男声清 | 女声清 |
|---|---|---|---|
| v1 | 20~120ms + 前后静音>0.15s | 9 | 12 |
| v2 | 放宽到 250ms + 0.08s + 浊音判据 | 18 | 5 |
| v3 | v2 + 向左右扩展到「峰值-12dB」 | 18（每处+40~60ms） | 5 |
| **v4** | **v2 + 扩展到两侧静音边界（整岛抹平）** | **18（每处 270~405ms）** | **5** |

v3→v4 的差别：`tools/gruntext53.py` 证明每个「嗯」的起音/收尾能量低于
`db > p35` 门槛被判成静音，v3 用「峰值-12dB」**猜**边界；v4 干脆把
夹在两段静音之间的**整个小岛**抹平，不用猜。

🔴 v4 加了「扩展后校验」：整岛抹平会把「嗯」前后的静音一起吃掉，若静音里
混着辅音，扩展后的 zcr 会明显升高。实测女声 3 处扩展后 zcr 从 <0.12 升到
0.18~0.21（包进摩擦音），这种自动回退到保守的 db 模式。

## 全篇实测（唯一变量 = 清理版本）

| 指标 | 男 原始 | 男 v3 | 男 v4 |
|---|---|---|---|
| click % | 0.0172 | 0.0172 | 0.0172 |
| 停顿中位 | 0.34s | 0.34s | 0.34s |
| 质心 | 793 | 792 | 781 |
| cps | 8.20 | 8.20 | 8.20 |
| 时长 | 560.1s | 560.1s | 560.1s |

女声 v3/v4：click / 停顿 / 质心 / cps 全部与原始一致。

## 🔴 本轮关键修正：片段必须包含「嗯」

z27 统一取 60~120s，但**女声的 5 个「嗯」在 214/351/387/389s**，
片段里一个都没有 —— 员工根本听不到差异，这是 z27 投 B=B 的可能原因之一。

本轮按「嗯」的实际位置选片段：
    男声 80~170s（含 85.09 / 114.40 / 133.55 / 134.12 / 155.86 共 5 个）
    女声 340~410s（含 351.07 / 387.42 / 389.79 共 3 个）

## 题目（零 EL，4 段单段三档评分）

男 v3 / 男 v4 / 女 v3 / 女 v4。
与 z27（v2，同题型）的分数直接对比 → 看 v2→v3→v4 是否递增。
若 v4 > v3 → v4 判据接进后处理链路；若持平 → 说明已到顶，v3 就够。

字段名必须与 app/ui/feedback.html 对齐：title / desc / dim / options[].key。
"""
import json
import os
import subprocess
import sys
import tempfile

FB = "/mnt/tts/fb"
VER = "v3.0"
ROUND = "z28"
TARGET_LUFS = -20.0

# (输出名, 源, 性别, 角色, 标签, 起点s, 长度s)
FILES = [
    ("z28_m_v3.wav", "/mnt/tts/diag/gruntclean53/v3/m_clean.wav",
     "male", "v3", "男声·v3扩展12dB(18处)", 80.0, 90.0),
    ("z28_m_v4.wav", "/mnt/tts/diag/gruntclean53/v4/m_clean.wav",
     "male", "v4", "男声·v4整岛抹平(18处)", 80.0, 90.0),
    ("z28_f_v3.wav", "/mnt/tts/diag/gruntclean53/v3/f_clean.wav",
     "female", "v3", "女声·v3扩展12dB(5处)", 340.0, 70.0),
    ("z28_f_v4.wav", "/mnt/tts/diag/gruntclean53/v4/f_clean.wav",
     "female", "v4", "女声·v4整岛抹平(5处)", 340.0, 70.0),
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
    """响度对齐。🔴 src 与 dst 相同会炸，内部走临时文件。"""
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
        key = name.replace("z28_", "").replace(".wav", "")
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
    out = os.path.join(FB, "items_z28.json")
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
    print("\n切换命令：cp %s/items_z28.json %s/items.json" % (FB, FB))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
