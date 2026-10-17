# -*- coding: utf-8 -*-
"""mk_z27.py —— z27 批次：验证「彻底清理极短填充音」有没有消除残留。

## 上一轮（z26）员工反馈
> 「整体有改善，但是没清理干净，说完话就跟带口头禅似的有尾音嗯」
> 「女声比男声好点」

两人都投 B=B（清理前/清理后同分），但文字反馈说有残留。

## 为什么没清干净（`tools/gruntaudit53.py` 审计）

原判据（v1）漏了两类「嗯」：
    · 时长 120~250ms 的（原上限卡在 120ms）
    · 前后静音不足 0.15s 的（原门槛 0.15s）
男声原判据只抓 9 个；放宽到 250ms+0.08s 后有 58 个。

但直接全放宽会误伤：`tools/gruntdiff53.py` 显示多出的 49 个里
只有 13 个是浊音「嗯」（zcr 0.038~0.084、300Hz 以下能量 0.46~0.85），
其余 36 个 zcr 高/低频低，是正常语音碎片，删了会切碎词语。

## v2 判据（三者同时满足）
    · 时长 20~250ms
    · 前后静音 > 0.08s
    · **浊音**：过零率 < 0.12 且 300Hz 以下能量占比 > 0.45
    · 峰值 > -32dB

在 EL 上验证：EL 男声同判据 16 个、女声 2 个 —— 说明 EL 也有这类
短浊音段（正常韩语短音节/句末鼻音），所以不清零到 0，只清我们多出来的。

## 全篇实测（唯一变量 = 清理判据版本）

| 指标 | 男 原始 | 男 v1清(9个) | 男 v2清(18个) |
|---|---|---|---|
| click % | 0.0172 | 0.0170 | 0.0172 |
| 停顿中位 | 0.34s | 0.34s | 0.34s |
| 质心 | 793 | 792 | 793 |
| cps | 8.20 | 8.20 | 8.20 |

女声：v2 清 5 个，click/停顿/质心/cps 全部不变。
→ v2 比 v1 多清 9 个（男），且不碰任何其他样本。

## 题目（零 EL）

4 段，男女各一组「v1 清理 / v2 清理」。
v1 是员工上一轮听过、反馈"没清理干净"的那版；v2 是多清了 9 个的版本。
若 v2 > v1 → 用 v2 判据接进后处理链路。

字段名必须与 app/ui/feedback.html 对齐：title / desc / dim / options[].key。
"""
import json
import os
import subprocess
import sys
import tempfile

FB = "/mnt/tts/fb"
VER = "v3.0"
ROUND = "z27"
CLIP = 60.0
START = 60.0
TARGET_LUFS = -20.0

FILES = [
    # (输出名, 源, 性别, 角色, 中文标签)
    ("z27_m_v1.wav", "/mnt/tts/diag/gruntclean53/m_clean.wav",
     "male", "clean_v1", "男声·清理v1(9个)"),
    ("z27_m_v2.wav", "/mnt/tts/diag/gruntclean53/v2/m_clean.wav",
     "male", "clean_v2", "男声·清理v2(18个)"),
    ("z27_f_v1.wav", "/mnt/tts/diag/gruntclean53/f_clean.wav",
     "female", "clean_v1", "女声·清理v1"),
    ("z27_f_v2.wav", "/mnt/tts/diag/gruntclean53/v2/f_clean.wav",
     "female", "clean_v2", "女声·清理v2"),
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
        key = name.replace("z27_", "").replace(".wav", "")
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
    out = os.path.join(FB, "items_z27.json")
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
    print("\n切换命令：cp %s/items_z27.json %s/items.json" % (FB, FB))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
