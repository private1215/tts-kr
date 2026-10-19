# -*- coding: utf-8 -*-
"""mk_z30.py —— z30 批次：验证「句首紧贴碎片清理 tailclean53c」的听感收益。

## 背景（19:45 复核 + 19:56 tailclean53c 上线）

EL 同文本对照（d172b8c4efb3）确认的差距：
    · 句末粘连尾巴：我们 49 vs EL 36 → tailclean53b 已清（m_tail3 / m_tail2_grunt）
    · 句首紧贴碎片（后间隔 20~50ms）：我们 11 vs EL 2（5.5 倍）
      → tailclean53c 清 10 处 → m_final.wav
    清理后：句首碎片 21→9（EL 12）；句末粘连保持 1；ASR 相似度 1.0000（没削字）。

## 题目（4 题，同窗口盲听，独立档位）

窗口 410~470s（60s）：含 53c 清理点 417.84/419.50/455.57 三处，
且 v4 在此窗口有句末尾巴约 6 处（411.68/413.29/420.13/431.87/438.14/466.00），
现象充足（铁律 11）。

    题1 男 v4           （最粗糙版，低锚）
    题2 男 v4+tail      （z29 已验证版本）
    题3 男 v4+tail+53c  （句首清理成品 m_final）
    题4 男 final+中频EQ （m_final + firequalizer +2dB，对齐 EL 音色）

题4 依据（measure53 同文本 EL 对照，确定因素）：
    中频1k-3k占比 5.76% vs EL 6.86% → +2dB 后 6.92%（对齐）
    同时 click 0.0172→0.0152（改善 12%），杂音仍优于 EL，ASR 内容一致。
    ⚠️ 本机 ffmpeg `equalizer` 滤镜实测失效，必须用 `firequalizer`。

决策规则（铁律 3：拿到答案改什么）：
    3 > 2 > 1 → 53c 加入生产链；3 < 2 → 回退
    4 > 3     → 中频 EQ 加入生产链；4 < 3 → EQ 回退

盲听：选项不标身份，只记服务端 note。
⚠️ z29 仍在收票（2/4 人），本轮只落 items_z30.json，不切换 items.json。
"""
import json
import os
import subprocess
import sys
import tempfile

FB = "/mnt/tts/fb"
VER = "v3.0"
ROUND = "z30"
TARGET_LUFS = -20.0

FILES = [
    ("z30_m_v4.wav", "/mnt/tts/diag/gruntclean53/v4/m_clean.wav",
     "male", "v4", "男声·v4整岛抹平", 410.0, 60.0),
    ("z30_m_tail.wav", "/mnt/tts/diag/tailclean53/m_tail2_grunt.wav",
     "male", "tail", "男声·v4+tail", 410.0, 60.0),
    ("z30_m_final.wav", "/mnt/tts/diag/tailclean53/m_final.wav",
     "male", "final", "男声·v4+tail+句首清理", 410.0, 60.0),
    ("z30_m_finaleq.wav", "/mnt/tts/diag/tailclean53/m_final_eq.wav",
     "male", "finaleq", "男声·final+中频EQ(+2dB)", 410.0, 60.0),
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
        key = name.replace("z30_", "").replace(".wav", "")
        t = os.path.join(FB, name)
        cut(src, t, start, clip)
        cur, after = norm(t, t)
        print("%s  LUFS %.1f -> %.1f" % (name, cur or float("nan"),
                                        after or float("nan")), flush=True)
        ITEMS.append({
            "id": "%s-%s-%s" % (VER, ROUND, key),
            "dim": "男声·整体自然度",
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
    out = os.path.join(FB, "items_z30.json")
    json.dump({"items": ITEMS}, open(out, "w"), ensure_ascii=False, indent=2)
    print("\n%s  %d 题（零 EL，未切换 items.json）" % (out, len(ITEMS)),
          flush=True)
    for i, it in enumerate(ITEMS, 1):
        print("  %d. %-24s %s" % (i, it["id"],
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
    print("\n切换命令（等 z29 收完票再执行）：cp %s/items_z30.json %s/items.json"
          % (FB, FB))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
