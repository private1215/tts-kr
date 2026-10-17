# -*- coding: utf-8 -*-
"""mk_z24.py —— z24 批次：**男声换 omnivoice 引擎**员工盲听裁决。

## 为什么出这题（z23 之后的实测）

z23 结论：变调/不变调员工完全听不出差别（都是 C，0 分），EL 判 B（1 分）。
连续三轮客观指标优化（click 0.0103→0.0041、音色 0.79→0.88）员工评分零变化
→ 按 S6 封存 click/EQ/音色/变调/停顿底部方向。

关键对照：**女声 omnivoice 得 1.5 分，男声 voxcpm 三轮都是 0 分**。
同样极简后处理思路女声有效男声无效 → 差别在引擎。

omnivoice 男声全篇实测（4594 字，同文案 d172b8c4efb3）：

| 指标 | omnivoice 男 | voxcpm 现役 v2.0 | EL |
|---|---|---|---|
| 音色 ×EL | **0.9557** | 0.7928 | 1.000 |
| click% | 0.0132 | 0.0041 | 0.0124 |
| 停顿中位 | 0.32s | 0.20s | 0.16s |
| CV | 0.64 | 0.66 | 0.70 |
| 质心 | 802 | 789 | 818 |
| cps | 8.31 | 8.37 | 8.27 |
| F0 | 176.7 | 187.2 | 200.7 |

→ 音色 +0.16 是质的飞跃，click/cps/质心天然对齐 EL，唯一短板是停顿 2 倍。

## 题目

**男声 4 段 + 女声 3 段**，同性别同文案同位置（60~120s），单段独立评分：

  男（d172b8c4efb3）：
    G  omnivoice 男声 + 仅响度对齐（极简，对齐女声 v1.1 路线）
    H  G + 停顿等比例压缩（scale 0.6，修 2 倍停顿）
    I  voxcpm 现役 v2.0（对照）
    J  EL 标杆

  女（00b6a45ad6aa）：
    K  omnivoice 女声现役（v1.1 极简成品）
    L  omnivoice 女声 + 响度对齐 + 停顿压缩（scale 0.8：现役 0.22s → EL 0.18s）
    M  EL 标杆

🔴 为什么必须男女都出（用户 2026-10-17 指出）：此前 z22/z23/z24 只出男声，
   理由是「男声 0 分是主战场」。但女声 1.5 分也没到 EL 的 2.0，
   且「omnivoice + 停顿压缩」这条新路对女声同样适用（现役 0.22s vs EL 0.18s）。
   只测一个性别既浪费员工时间，也让女声连续三轮无人跟进。

一轮同时回答四个问题：男声换引擎？男声换引擎+修停顿？
女声修停顿？男女与各自 EL 标杆的差距？

身份只记服务端 note，题干/选项不出现引擎名/版本/EL。

字段名必须与 app/ui/feedback.html 对齐：title / desc / dim / options[].key。
"""
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, "/tmp")

FB = "/mnt/tts/fb"
VER = "v2.0"
ROUND = "z24"
CLIP = 60.0
START = 60.0
TARGET_LUFS = -20.0

RAW_OMNI = "/mnt/tts/diag/omnim52/m_full_raw.wav"
VOX_CUR = "/mnt/tts/diag/l3_52/D_mild_no_clean.wav"
EL_M = "/mnt/tts/data/cases/d172b8c4efb3/merged.wav"
# 女声（不同文案，员工听不出关联）
RAW_OMNI_F = "/tmp/omniraw_f.wav"
OMNI_F_CUR = "/mnt/tts/output/v11_f_official/v11_f_official-1.mp3"
EL_F = "/mnt/tts/data/cases/00b6a45ad6aa/merged.wav"

SCALE_M = 0.6      # 男声停顿缩放（0.32s -> ~0.20s）
SCALE_F = 0.8      # 女声停顿缩放（0.22s -> ~0.18s，已接近 EL，温和压）
FLOOR = 0.20


def lufs(path):
    """ffmpeg ebur128 集成响度。⚠️ 统计用 info 级别，不能传 -v error。"""
    r = subprocess.run(["ffmpeg", "-v", "info", "-i", path,
                        "-af", "ebur128=framelog=quiet", "-f", "null", "-"],
                       capture_output=True, text=True, timeout=600)
    tail = [l for l in r.stderr.splitlines() if "I:" in l]
    if tail:
        return round(float(tail[-1].split("I:")[1].split()[0]), 1)
    return None


def to16k(src, dst):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-ar", "16000", "-ac", "1", dst], check=True)
    return dst


def norm(src, dst, target=TARGET_LUFS):
    cur = lufs(src)
    d = target - cur
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-af", "volume=%gdB" % d, "-ar", "16000", "-ac", "1", dst],
                   check=True)
    return cur, lufs(dst)


def compress(src, dst, scale=SCALE_M, floor=FLOOR):
    """调 pausecompress52.py 做等比例停顿缩放。"""
    r = subprocess.run([sys.executable, "/tmp/pausecompress52.py", src, dst,
                        "--scale", str(scale), "--floor", str(floor),
                        "--min", "0.25"], capture_output=True, text=True)
    print("  压缩(scale=%.2f): %s" % (scale, r.stdout.strip()[-90:]), flush=True)
    return r.returncode == 0


def cut(src, dst, start=START, clip=CLIP):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-ss", str(start), "-t", str(clip),
                    "-ar", "16000", "-ac", "1", dst], check=True)
    return dst


Q = "请听这一段，听完后回答：**这段听起来像不像真人在说话？**"
D = ("就当你在手机或车里偶然刷到这条股票广播，随便听，凭第一感觉选。\n"
     "不用分辨细节，也不用和专业主播比。\n"
     "下面三个选项是**评价档位**，不是几段不同的音频——只播一遍，选一个最贴近你感觉的。")

ITEMS = []


def build():
    tmpd = tempfile.mkdtemp()
    g16 = os.path.join(tmpd, "g16.wav")
    h16 = os.path.join(tmpd, "h16.wav")

    # G：omni 男声裸 → 16k → 仅响度对齐
    to16k(RAW_OMNI, g16)
    c1, c2 = norm(g16, os.path.join(FB, "z24_m_omni.wav"))
    print("z24_m_omni.wav  LUFS %.1f -> %.1f" % (c1, c2), flush=True)

    # H：G + 停顿等比例压缩 + 再对齐响度
    compress(g16, h16)
    c3, c4 = norm(h16, os.path.join(FB, "z24_m_omni_pause.wav"))
    print("z24_m_omni_pause.wav  LUFS %.1f -> %.1f" % (c3, c4), flush=True)

    # I：voxcpm 现役 v2.0（只做响度对齐，其它不动）
    c5, c6 = norm(VOX_CUR, os.path.join(FB, "z24_m_vox.wav"))
    print("z24_m_vox.wav  LUFS %.1f -> %.1f" % (c5, c6), flush=True)

    # J：EL 男声标杆
    c7, c8 = norm(EL_M, os.path.join(FB, "z24_m_el.wav"))
    print("z24_m_el.wav  LUFS %.1f -> %.1f" % (c7, c8), flush=True)

    # ---- 女声（不同文案）----
    k16 = os.path.join(tmpd, "k16.wav")
    l16 = os.path.join(tmpd, "l16.wav")

    # K：女声现役成品（v1.1 极简）只做响度对齐
    to16k(OMNI_F_CUR, k16)
    c9, c10 = norm(k16, os.path.join(FB, "z24_f_cur.wav"))
    print("z24_f_cur.wav  LUFS %.1f -> %.1f" % (c9, c10), flush=True)

    # L：女声裸 + 响度对齐 + 停顿压缩（scale 0.8，0.22s -> EL 0.18s）
    to16k(RAW_OMNI_F, l16)
    compress(l16, os.path.join(tmpd, "l16c.wav"), SCALE_F, 0.15)
    c11, c12 = norm(os.path.join(tmpd, "l16c.wav"),
                    os.path.join(FB, "z24_f_pause.wav"))
    print("z24_f_pause.wav  LUFS %.1f -> %.1f" % (c11, c12), flush=True)

    # M：EL 女声标杆
    c13, c14 = norm(EL_F, os.path.join(FB, "z24_f_el.wav"))
    print("z24_f_el.wav  LUFS %.1f -> %.1f" % (c13, c14), flush=True)

    # 统一切 60~120s
    ALL = ("z24_m_omni.wav", "z24_m_omni_pause.wav", "z24_m_vox.wav",
           "z24_m_el.wav", "z24_f_cur.wav", "z24_f_pause.wav", "z24_f_el.wav")
    for f in ALL:
        p = os.path.join(FB, f)
        t = os.path.join(tmpd, "t.wav")
        cut(p, t)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", t,
                        "-ar", "16000", "-ac", "1", p], check=True)

    FILES = [
        ("z24_m_omni.wav", "omni_minimal", "omnivoice 男声·仅响度"),
        ("z24_m_omni_pause.wav", "omni_pause", "omnivoice 男声·响度+停顿压缩"),
        ("z24_m_vox.wav", "vox_current", "voxcpm 现役"),
        ("z24_m_el.wav", "el", "EL 标杆"),
        ("z24_f_cur.wav", "omni_f_current", "omnivoice 女声·现役"),
        ("z24_f_pause.wav", "omni_f_pause", "omnivoice 女声·响度+停顿压缩"),
        ("z24_f_el.wav", "el_f", "EL 标杆"),
    ]
    for name, role, label in FILES:
        key = name.replace("z24_", "").replace(".wav", "")
        gender = "女" if name.startswith("z24_f_") else "男"
        ITEMS.append({
            "id": "%s-%s-%s" % (VER, ROUND, key),
            "dim": "%s声·整体自然度" % gender,
            "title": Q,
            "desc": D,
            "ref": None,
            "note": "性别=%s 角色=%s 源=%s 位置=%.0fs 长度=%.0fs" % (
                "female" if gender == "女" else "male", role, label,
                START, CLIP),
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
    out = os.path.join(FB, "items_z24.json")
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
                    print("    ❌ 文件缺失或过小（应约 %.0f 字节）"
                          % (16000 * 60 * 2), flush=True)
                    ok = False
    print("  %s" % ("✅ 字段/文件自检全过" if ok else "❌ 自检未过，勿切换"))
    print("\n切换命令：cp %s/items_z24.json %s/items.json" % (FB, FB))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
