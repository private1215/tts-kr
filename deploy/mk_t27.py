#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第二十七轮（t）题库：**闭环第④条 —— 验证换引擎后有没有真的变好**。

## 这一轮存在的唯一理由
第 26 轮闭环已完成 ①②③：
  ① 3 份答卷完全一致（项目至今一致性最高）
  ② 每题对应动作 = **男声引擎 voxcpm → omnivoice**（已执行）
  ③ 动作已验证（553 秒真长文 Σz²=166，4k~8k 极差 94.9%→81.2%）
→ 现在必须补 **④：新题库已更新，且新题只问「有没有变好」**。

🔴 **本轮不问任何已排除的维度**（闭环台账 fb_loop.py 记录在案）：
   静态频谱 / 语速 / 候选EQ / 合成底噪 / MP3编码 / 断句停顿 / 情绪 /
   前后位置 / 字间清晰度 —— 全部已排除，不再重复问。

## 只有三道题，每道都指向一个动作
| 题 | 答案对应的动作 |
|---|---|
| t1 新引擎 vs 旧引擎直接比 | 听得出改善 → 换引擎成功；听不出 → 查 EQ 是否失配 |
| t2 「电子感」还在吗 | 还在 → 是 OmniVoice 的架构限制，换引擎不够 |
| t3 能否直接发客户 | **长期基线锚点**（此前 3 轮 13 票全选 EL） |

## 素材（用户铁律：全部取自 ≥180s 真长文）
- 新引擎：/mnt/tts/output/longprobe-omni-6cdd/（**553 秒**，刚生成的）
- 旧引擎：/mnt/tts/output/ab-rt-base/（**496 秒**，切换前的现役）
- EL 标杆：/mnt/tts/data/cases/6cdd25bc75f2/merged.wav（509 秒）
三段都是同一份文案，能直接对照。
"""
import hashlib
import json
import os
import shutil
import subprocess

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
TMPD = "/mnt/tts/diag/_t27"
TGT_LUFS = -26.1
DUR = 30.0

NEW_ = "/mnt/tts/output/longprobe-omni-6cdd/longprobe-omni-6cdd-1.mp3"  # 553s 新
OLD_ = "/mnt/tts/output/ab-rt-base/ab-rt-base-1.mp3"                   # 496s 旧
EL_S = "/mnt/tts/data/cases/6cdd25bc75f2/merged.wav"                 # 509s EL

# 起点互不重叠（30s 窗口，间隔 ≥30s），全在 496s 内（三段都能取到）
P_E, P_M, P_L = 60.0, 200.0, 380.0


def lufs(p):
    r = subprocess.run(["ffmpeg", "-i", p, "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    v = [l for l in r.stderr.splitlines() if "I:" in l and "LUFS" in l]
    return float(v[-1].split("I:")[1].split("LUFS")[0].strip()) if v else None


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()[:8]


def cut(src, start, dur):
    t = os.path.join(TMPD, "_t%d.wav" % int(start * 1000))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-ss", "%.1f" % start, "-ar", "16000", "-ac", "1",
                    "-t", "%.1f" % dur, t], check=True)
    got = os.path.getsize(t) / 2.0 / 16000
    if got < dur - 0.5:
        raise SystemExit("🔴 切片越界：%s 第%.0fs起取%.0fs，实际仅%.1fs"
                         % (os.path.basename(src), start, dur, got))
    return t


def emit(wav, name, title):
    g = TGT_LUFS - lufs(wav)
    mp3 = os.path.join(FB_DIR, "%s.mp3" % name)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav,
                    "-af", "volume=%gdB" % g,       # 🔴 dB 不是 %gdBB
                    "-b:a", "192k", "-metadata", "title=%s" % title, mp3],
                   check=True)
    back = lufs(mp3)
    if back is None:
        print("  %-12s 🔴 读不出 LUFS" % name)
        return None
    print("  %-12s md5=%s  LUFS %6.1f %s"
          % (name, md5(mp3), back,
             "OK" if abs(back - TGT_LUFS) < 1.0 else "!!爆表"))
    return mp3


def main():
    os.makedirs(TMPD, exist_ok=True)
    made = {}
    for tag, src in (("new", NEW_), ("old", OLD_), ("el", EL_S)):
        for pos, st in (("e", P_E), ("m", P_M), ("l", P_L)):
            nm = "t27_%s_%s" % (tag, pos)
            made[nm] = emit(cut(src, st, DUR), nm, "clip %s" % nm)

    ctrl = cut(NEW_, 150.0, DUR)
    made["t27_ctrl1"] = emit(ctrl, "t27_ctrl1", "ctrl A")
    made["t27_ctrl2"] = emit(ctrl, "t27_ctrl2", "ctrl B")

    print("\n=== md5 自检 ===")
    inv = {}
    for nm, p in made.items():
        inv.setdefault(md5(p), []).append(nm)
    bad = False
    for h, ns in inv.items():
        if len(ns) > 1:
            ok = set(ns) <= {"t27_ctrl1", "t27_ctrl2"}
            print("  %-9s %s" % (h, "OK(控制题同素材)" if ok
                                 else "🔴 撞车 -> %s" % ns))
            if not ok:
                bad = True
    if not bad:
        print("  ✅ 无撞车")

    items = [
        # ---- 1. 直接验证「换引擎有没有用」----
        {"id": "t1_new_vs_old_20261015t", "dim": "换引擎验证",
         "title": "这两段，哪一段更自然？",
         "desc": "两段都是男声，同一段文案，各 30 秒。\n\n"
                 "**第 1 段是刚换的新引擎，第 2 段是换之前的版本。**",
         "ref": None,
         "note": "🔴 闭环验证核心题。\n"
                 "A=新引擎(OmniVoice) B=旧引擎(VoxCPM2)。\n"
                 "机器侧已改善：4k~8k 沿时间轴极差 94.9%→81.2%、跨窗Σz² 中位 783→187。\n"
                 "→ 这题问**耳朵能不能听出这个改善**。\n"
                 "选C(差不多) → 换引擎在听感上没兑现，得查男声 EQ 是否失配"
                 "（那段EQ是按 VoxCPM2 频谱拟合的）。",
         "options": [
             {"key": "A", "file": "t27_new_m.mp3", "label": "第 1 段（换之后）更自然"},
             {"key": "B", "file": "t27_old_m.mp3", "label": "第 2 段（换之前）更自然"},
             {"key": "C", "file": None, "label": "差不多，听不出差别"}]},

        # ---- 2. 「电子感」这个核心问题解决了吗 ----
        {"id": "t2_electronic_now_20261015t", "dim": "电子感（复验）",
         "title": "这段现在听起来，像真人录的还是像电子合成的？",
         "desc": "一段男声，30 秒，是**换引擎之后的新版本**。\n\n"
                 "就按「像真人还是像电子合成」判断。",
         "ref": {"label": "▶ 点这个播放", "file": "t27_new_m.mp3",
                 "note": "换引擎之后的版本。"},
         "note": "🔴 第26轮 s1 拿到 3/3「完全不能接受，一听就是合成的」，"
                 "换引擎就是为了解决它。\n"
                 "→ 这题复验：**换引擎后到底解决没解决**。\n"
                 "若仍选「像电子合成」→ 是 OmniVoice 的架构限制，"
                 "换引擎这条路也走不通，得考虑非自回归方案或微调。",
         "options": [
             {"key": "A", "file": None, "label": "① 像真人录的"},
             {"key": "B", "file": None, "label": "② 有点合成感，但能接受"},
             {"key": "C", "file": None, "label": "③ 还是明显像电子合成"}]},

        # ---- 3. 长期基线锚点 ----
        {"id": "t3_send_baseline_20261015t", "dim": "可发送性",
         "title": "这 30 秒，哪一段可以直接发给客户？",
         "desc": "两段都是男声，同一段文案，各 30 秒。\n\n"
                 "按「能不能直接发给客户」判断。",
         "ref": None,
         "note": "**长期基线锚点。**\n"
                 "此前 3 轮共 13 张票**无一例外全选 EL**。\n"
                 "→ 换引擎后第一次接受检验：\n"
                 "   若首次出现「我们也能发」→ 换引擎真的有效。",
         "options": [
             {"key": "A", "file": "t27_new_l.mp3", "label": "第 1 段（换引擎之后）"},
             {"key": "B", "file": "t27_el_l.mp3", "label": "第 2 段（EL）"},
             {"key": "C", "file": None, "label": "都不行"}]},
    ]

    items.append(
        {"id": "ctrl_pace_20261015t", "dim": "答题有效性校验",
         "title": "【最后一题】这两段，哪一段念得更快？",
         "desc": "**这道题必须选一个**（选错也没关系）。\n\n"
                 "同一段音频放了两遍，各 30 秒。听 5 秒就够。",
         "ref": None,
         "note": "隐藏控制题：正解=C 一样快。选 A/B 说明没在听，整份作废。",
         "options": [
             {"key": "A", "file": "t27_ctrl1.mp3", "label": "第 1 段念得更快"},
             {"key": "B", "file": "t27_ctrl2.mp3", "label": "第 2 段念得更快"},
             {"key": "C", "file": None, "label": "两段一样快"}]})

    print("\n=== 题目完整性 ===")
    empty = []
    for it in items:
        hf = any(o.get("file") for o in it.get("options") or [])
        hr = bool(it.get("ref") and it["ref"].get("file"))
        if not (hf or hr):
            empty.append(it["id"])
        print("  %-28s 选项=%s 参考=%s"
              % (it["id"], "有" if hf else "无", "有" if hr else "无"))
    print("  ✅ 每题都有音频" if not empty else "  🔴 废题: %s" % empty)

    p = os.path.join(FB_DIR, "items.json")
    if os.path.exists(p):
        shutil.copyfile(p, p + ".bak-before-t")
        print("\n已备份旧题库 ->", p + ".bak-before-t")
    json.dump({"items": items}, open(p, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("题库已写入 %s  共 %d 题" % (p, len(items)))
    if bad or empty:
        print("🔴 存在拦截项，先别让员工答题")
    print("\n发布后确认：curl -s http://服务器:8000/api/fb/where")
    print("归档：python3 tools/fb_loop.py --archive t")


if __name__ == "__main__":
    main()