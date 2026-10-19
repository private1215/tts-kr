# -*- coding: utf-8 -*-
"""gen_z33f.py —— 给 z32 批次补 2 道**女声**题（旧链 vs 新链）。

## 为什么补
v3.3 把女声接进了 EQ+clickfix，L3 客观六项全改善。但员工还没听过，
**客观变好不等于人耳觉得好**（z44：13dB 物理改动，9 票仅 1 票指向修后）。
而 z32 批次只有男声题 → 员工回来时没有女声素材可判。
votes.jsonl 最后一条是 10-17，**z32 零投票**，可以安全扩批。

## 题型（照抄 z32 男声题，保证员工看到的是同一套界面）
每题只播一段音频 + 3 个**评价档位**（不是 A/B 二选一，反馈台 UI 只支持这个）：
    ① 像真人，很自然  ② 听得出是 AI，但还算舒服  ③ 明显是机器，听着别扭
两道题分别放「女声旧链」「女声新链」，比较两者的档位分布即可判断改动方向。

## 素材
    z33_f_old.wav ← l3f53-1.mp3  390s 起 60s（v3.2 女声链，修前）
    z33_f_new.wav ← l3f53b-1.mp3 390s 起 60s（v3.3 女声链，修后）
两条同文案、同引擎、同 cps 锚，但时长差 0.34s → **必须过 alignchk48**（铁律9）。

## 铁律自检
  1. alignchk48 相似度 ≥ 0.55（同一句，否则员工听到的差别只是内容不同）
  2. md5 记录
  3. LUFS 回读（确认切片没被改响度；z32 男声题是 -25.3 LUFS 原生电平）
  4. `/api/fb/where` missing 必须空
  5. 🔴 盲听铁律：选项不许标身份，身份只写服务端 `note`
  6. 🔴 只让第一个 option 带 file（否则每个选项都渲染播放器）

用法: venv/bin/python -u /mnt/tts/diag/gen_z33f.py [--apply]
"""
import hashlib
import json
import os
import subprocess
import sys

PY = "/mnt/tts/projects/qwen3-tts/venv/bin/python"
T = "/mnt/tts/projects/qwen3-tts/tools"
FB = "/mnt/tts/fb"
ITEMS = os.path.join(FB, "items.json")
OLD_SRC = "/mnt/tts/output/l3f53/l3f53-1.mp3"
NEW_SRC = "/mnt/tts/output/l3f53b/l3f53b-1.mp3"
POS, DUR = 390.0, 60.0


def run(a, **kw):
    print("$ " + " ".join(a), flush=True)
    return subprocess.run(a, check=True, **kw)


def cut(src, dst):
    run(["ffmpeg", "-y", "-v", "error", "-ss", str(POS), "-t", str(DUR),
         "-i", src, "-ar", "44100", "-ac", "1", dst])


def norm_lufs(paths, target=None):
    """把多条素材用**静态音量**统一到同一 LUFS。

    为什么必须做：EQ 抬中频会把积分响度一起抬起来（实测旧链 -22.7 / 新链 -22.3）。
    A/B 时人耳会系统性偏向更响的那条 → 不统一响度，投票测的是响度不是音色。
    🔴 用 `volume=%gdB`，**不是 `%gdBB`**（后者 ffmpeg 直接报错）。
    """
    vals = [lufs(p) for p in paths]
    if target is None:
        target = sum(float(v.split()[0]) for v in vals) / len(vals)
    print("  统一响度到 %.1f LUFS" % target, flush=True)
    for p, v in zip(paths, vals):
        d = target - float(v.split()[0])
        tmp = p + ".tmp.wav"
        run(["ffmpeg", "-y", "-v", "error", "-i", p, "-af",
             "volume=%gdB" % d, "-ar", "44100", "-ac", "1", tmp])
        os.replace(tmp, p)
        print("    %s  %s → %s（%+.1f dB）"
              % (os.path.basename(p), v, lufs(p), d), flush=True)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def lufs(p):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", p, "-af",
                        "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=summary",
                        "-f", "null", "-"], capture_output=True, text=True)
    for ln in (r.stderr or "").splitlines():
        if "Input Integrated" in ln:
            return ln.split(":")[1].strip()
    return "?"


def main():
    apply = "--apply" in sys.argv
    old_w = os.path.join(FB, "z33_f_old.wav")
    new_w = os.path.join(FB, "z33_f_new.wav")
    cut(OLD_SRC, old_w)
    cut(NEW_SRC, new_w)
    print("\n===== 铁律0：统一响度（消除「人偏向更响」的混杂）=====", flush=True)
    norm_lufs([old_w, new_w])
    print("\n===== 铁律1：文本对齐（alignchk48，须 ≥0.55）=====", flush=True)
    run([PY, "-u", "/mnt/tts/diag/alignchk48.py", old_w, new_w])
    print("\n===== 铁律2/3：md5 + LUFS 回读 =====", flush=True)
    for p in (old_w, new_w):
        print("  %s  md5=%s  LUFS=%s" % (os.path.basename(p), md5(p), lufs(p)),
              flush=True)

    j = json.load(open(ITEMS, encoding="utf-8"))
    have = {it["id"] for it in j["items"]}
    new_items = [
        {
            "id": "v3.3-z32-f_v32",
            "dim": "女声·整体自然度（v3.2 女声链）",
            "title": "请听这一段，听完后回答：**这段听起来像不像真人在说话？**",
            "desc": ("就当你在手机或车里偶然刷到这条股票广播，随便听，凭第一感觉选。\n"
                     "不用分辨细节，也不用和专业主播比。\n"
                     "下面三个选项是**评价档位**，不是几段不同的音频——只播一遍，"
                     "选一个最贴近你感觉的。"),
            "ref": None,
            "note": ("性别=female 角色=v3.2女声链 源=아모텍/female_full.txt 3415字 "
                     "位置=390s 长度=60s 因子=基线 "
                     "客观=中频0.88 click1.24x 质心0.97 杂音1.27 音色0.9204"),
            "ver": "v3.3",
            "options": [
                {"key": "A", "label": "① 像真人，很自然", "file": "z33_f_old.wav"},
                {"key": "B", "label": "② 听得出是 AI，但还算舒服",
                 "file": None, "text": "（同上，只是评价档位）"},
                {"key": "C", "label": "③ 明显是机器，听着别扭",
                 "file": None, "text": "（同上，只是评价档位）"},
            ],
        },
        {
            "id": "v3.3-z32-f_v33",
            "dim": "女声·整体自然度（v3.3 女声链）",
            "title": "请听这一段，听完后回答：**这段听起来像不像真人在说话？**",
            "desc": ("就当你在手机或车里偶然刷到这条股票广播，随便听，凭第一感觉选。\n"
                     "不用分辨细节，也不用和专业主播比。\n"
                     "下面三个选项是**评价档位**，不是几段不同的音频——只播一遍，"
                     "选一个最贴近你感觉的。"),
            "ref": None,
            "note": ("性别=female 角色=v3.3女声链 源=아모텍/female_full.txt 3415字 "
                     "位置=390s 长度=60s 因子=中频EQ+2dB+clickfix0.26 "
                     "客观=中频1.01 click0.00x 质心0.99 杂音1.12 音色0.9227"),
            "ver": "v3.3",
            "options": [
                {"key": "A", "label": "① 像真人，很自然", "file": "z33_f_new.wav"},
                {"key": "B", "label": "② 听得出是 AI，但还算舒服",
                 "file": None, "text": "（同上，只是评价档位）"},
                {"key": "C", "label": "③ 明显是机器，听着别扭",
                 "file": None, "text": "（同上，只是评价档位）"},
            ],
        },
    ]
    add = [it for it in new_items if it["id"] not in have]
    if not add:
        print("\n两道女声题都已在题库里，跳过写入")
    else:
        print("\n待追加 %d 题：%s" % (len(add), [it["id"] for it in add]))
        if not apply:
            print("（未加 --apply，不写 items.json）")
            return 0
        bak = ITEMS + ".bak-before-z33f"
        if not os.path.exists(bak):
            import shutil
            shutil.copy2(ITEMS, bak)
        j["items"].extend(add)
        with open(ITEMS, "w", encoding="utf-8") as f:
            json.dump(j, f, ensure_ascii=False, indent=2)
        print("✅ 已写入 %s（备份 %s）" % (ITEMS, bak))

    print("\n===== 铁律4：/api/fb/where 自检 =====", flush=True)
    r = subprocess.run(["curl", "-s", "http://127.0.0.1:8000/api/fb/where"],
                       capture_output=True, text=True)
    print(r.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
