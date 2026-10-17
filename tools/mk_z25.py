# -*- coding: utf-8 -*-
"""mk_z25.py —— z25 批次：验证 **v3.0 线上链路**（男声 omnivoice）的实际效果。

## 为什么还要再测一轮

z24 测的是**手工制作的极简版**（omni 裸输出 + 我本地脚本响度对齐），
员工给 1.5 分。但线上 v3.0 链路与那版仍有差异，必须验证员工听到的
**真实线上产物**是否也是 1.5 分：

| 差异点 | z24 手工版 | v3.0 线上 |
|---|---|---|
| 响度 | -20 LUFS（题目统一） | -25.0 LUFS（fix_chain DEFAULT_LUFS 男） |
| 采样/编码 | 16k wav | 24k -> 16k 重采样 + mp3 192k |
| 语速锚 | 无 | target-cps 8.27（atempo 轻微变速） |
| 交付 | wav | mp3 |

另外这轮同时把**女声 v3.0 线上产物**一起测（男女都要有，见 z24 教训）。

## 题目（**只有我们的版本，不放 EL**）

🔴 2026-10-17 用户批评后的修正：
   首版 4 题里有 2 题是 EL（一半题目在测 EL），我当时的"改进"只是把顺序
   从 `[我们男, 我们女, EL男, EL女]` 打乱成 `[EL女, 女v3.0, 男v3.0, EL男]`
   —— **EL 一段没删，员工听的内容一模一样**，纯粹是脱裤子放屁。
   已立规矩：**员工时间 100% 花在给我们反馈上**。

   本项目没有真人原声（案例库 `source=elevenlabs`，merged 就是 EL 生成），
   EL 只是唯一锚点。但**已有 5 轮 EL 基线**（z20~z24，员工听过、稳定在 1~2 分），
   继续每轮放它是重复测已知信息。
   → **z25 起：默认不放 EL**，只测我们自己的版本，评分直接与历史基线比。
   → 若某轮确实需要重新校准锚点，最多放 **1 段**，且要在记忆里写明理由。

2 段，男女各一，同位置（60~120s），单段独立评分：
  男声 v3.0 线上成品（l3-m-omni，4594 字）
  女声 v3.0 线上成品（l3-f-omni，3417 字）

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
# 🔴 固定种子打乱题目顺序：EL 不能集中在末尾（员工会猜到「后两题是特殊材料」，
#    反而污染前两题的判断）。固定种子保证可复现——同一个种子永远同一个顺序。

FILES = [
    # (输出名, 源, 性别, 角色, 中文标签)
    ("z25_m_v30.wav", "/mnt/tts/output/l3-m-omni/l3-m-omni-1.mp3",
     "male", "v30_male", "男声 v3.0 线上成品"),
    ("z25_f_v30.wav", "/mnt/tts/output/l3-f-omni-2/l3-f-omni-1.mp3",
     "female", "v30_female", "女声 v3.0 线上成品"),
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
    # 只有我们自己的版本、男女不同文案，不需要打乱顺序（没有身份可猜）
    print("\n题目顺序：", flush=True)
    for i, it in enumerate(ITEMS, 1):
        print("  %d. %-22s %s" % (i, it["id"], it["note"].split("源=")[-1]),
              flush=True)

    out = os.path.join(FB, "items_z25.json")
    json.dump({"items": ITEMS}, open(out, "w"), ensure_ascii=False, indent=2)
    print("\n%s  %d 题" % (out, len(ITEMS)), flush=True)

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
