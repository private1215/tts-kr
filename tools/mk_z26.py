# -*- coding: utf-8 -*-
"""mk_z26.py —— z26 批次：验证优化点「**移除极短填充音**」有没有改善。

## 优化点来源（员工明确指出，不是我自己猜的）

z25 员工反馈原话：
> 「第一道题 A 特别特别短的 嗯 超级短 存在」
> 「第二道题 A 特别特别短的 嗯 超级短 存在」

离线定位（`tools/grunt53b.py`，16k 统一口径）：

    裸输出 yohan 参考   12 个 (1.30/分钟)
    裸输出 EL 参考       3 个 (0.33/分钟)
    EL参考+fixchain      5 个 (0.54/分钟)
    现役成品 mp3         9 个 (0.96/分钟)
    男 EL 标杆           **0 个**

→ 「嗯」是 OmniVoice 模型自发生成（不是后处理引入：裸输出 12 → fixchain 后 9）。
→ 换 EL 参考能减 50-75% 但消不掉；EL 本身是 0。
→ 员工听得见，且与她说了 N 次的「断句有异声」吻合。

## 修复（`tools/gruntclean53.py`）

检测「夹在两个长停顿之间的极短孤立发声段」（20~120ms，前后静音 >0.15s），
用**等长静音**替换（总时长不变），5ms 淡入淡出避免 click。

为什么阈值卡 120ms：韩语正常短音节（네/예/어）通常 >150ms，
「20~120ms 且前后都是 >0.15s 静音」在正常语流中几乎不出现。

## 全篇实测（460~560s，同链路同文本，唯一变量=是否清理）

| 指标 | 男 清理前 | 男 清理后 | 男 EL |
|---|---|---|---|
| 极短嗯 | 9 | **0** | 0 |
| click % | 0.0172 | 0.0170 | 0.0124 |
| 停顿中位 | 0.34s | 0.34s | 0.16s |
| 质心 | 793 | 792 | 818 |
| cps | 8.20 | 8.20 | 8.27 |

女声：12 → 0，其余同样不变。

→ **只去掉那一声「嗯」，其他样本一个都没动。**

## 题目（零 EL，员工时间全花在优化点上）

4 段，男女各一组「清理前 / 清理后」，同位置 60~120s。
若清理后 > 清理前 → 上线（把它接进 fix_chain 或 server.py 后处理）。

字段名必须与 app/ui/feedback.html 对齐：title / desc / dim / options[].key。
"""
import json
import os
import subprocess
import sys
import tempfile

FB = "/mnt/tts/fb"
VER = "v3.0"
ROUND = "z26"
CLIP = 60.0
START = 60.0
TARGET_LUFS = -20.0

FILES = [
    # (输出名, 源, 性别, 角色, 中文标签)
    ("z26_m_before.wav", "/mnt/tts/output/l3-m-omni/l3-m-omni-1.mp3",
     "male", "before", "男声·清理前"),
    ("z26_m_after.wav", "/mnt/tts/diag/gruntclean53/m_clean.wav",
     "male", "after", "男声·清理后"),
    ("z26_f_before.wav", "/mnt/tts/output/l3-f-omni-2/l3-f-omni-1.mp3",
     "female", "before", "女声·清理前"),
    ("z26_f_after.wav", "/mnt/tts/diag/gruntclean53/f_clean.wav",
     "female", "after", "女声·清理后"),
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
        key = name.replace("z26_", "").replace(".wav", "")
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
    out = os.path.join(FB, "items_z26.json")
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
                if not e or sz < 16000 * 60 * 2 // 2:
                    print("    ❌ 文件缺失或过小")
                    ok = False
    print("  %s" % ("✅ 字段/文件自检全过" if ok else "❌ 自检未过，勿切换"))
    print("\n切换命令：cp %s/items_z26.json %s/items.json" % (FB, FB))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
