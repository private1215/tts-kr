#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第二十六轮（s）题库：**每道题都必须能触发一个动作**。

🔴 2026-10-15 用户纠正（本脚本的设计前提）：
> **「反馈的目的是为了更好的优化语音，不要为了反馈而反馈」**

所以出题前先自问一句：**「拿到答案之后，我要改什么？」**
答不出来 → 这道题不该出。

据此删掉的题（都是"为了反馈而反馈"）：
  ×「差距大不大」    → 答完没有任何动作，纯信息收集
  ×「哪个引擎最接近 EL」→ 那是技术选型，不该让员工替我们选
  ×「多引擎排序」     → 5 个选项无对照，且引擎清单我们本来就知道

本轮只留 4 道实质题，每道对应一个明确决策：

| 题 | 拿到答案后的动作 |
|---|---|
| s1 能不能接受这个程度 | 决定**要不要为音质继续投入**（换引擎 vs 带着缺陷走） |
| s2 最想先改哪一件 | 决定**下一阶段的工作顺序** |
| s3 哪个引擎更能发 | 决定**男声换不换引擎**（素材就绪才出） |
| s4 能否直接发客户 | **长期基线锚点**（连续三轮 13 票全选 EL，出现变化才算有进展） |

## 素材
全部取自 ≥180s 长文案（用户铁律：短文案验收没意义）。
"""
import hashlib
import json
import os
import shutil
import subprocess

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
TMPD = "/mnt/tts/diag/_s26"
TGT_LUFS = -26.1
DUR = 30.0

CUR = "/mnt/tts/output/ab-rt-base/ab-rt-base-1.mp3"     # 现役 496s
EL_S = "/mnt/tts/data/cases/6cdd25bc75f2/merged.wav"   # EL 标杆 509s
ENG_D = "/mnt/tts/output"

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


def first_mp3(d):
    if not os.path.isdir(d):
        return None
    for f in sorted(os.listdir(d)):
        if f.endswith(".mp3"):
            return os.path.join(d, f)
    return None


def main():
    os.makedirs(TMPD, exist_ok=True)
    made = {}

    for tag, src in (("cur", CUR), ("el", EL_S)):
        for pos, st in (("e", P_E), ("m", P_M), ("l", P_L)):
            nm = "s26_%s_%s" % (tag, pos)
            made[nm] = emit(cut(src, st, DUR), nm, "clip %s" % nm)

    # 引擎对照素材（哪个跑出来取哪个）
    omni_src = first_mp3(os.path.join(ENG_D, "eng-omnivoice"))
    if omni_src:
        try:
            made["s26_eng_omnivoice_m"] = emit(
                cut(omni_src, 20.0, DUR), "s26_eng_omnivoice_m", "eng omni")
        except SystemExit as e:
            print("  %s" % e)

    ctrl = cut(CUR, 150.0, DUR)
    made["s26_ctrl1"] = emit(ctrl, "s26_ctrl1", "ctrl A")
    made["s26_ctrl2"] = emit(ctrl, "s26_ctrl2", "ctrl B")

    print("\n=== md5 自检 ===")
    inv = {}
    for nm, p in made.items():
        inv.setdefault(md5(p), []).append(nm)
    bad = False
    for h, ns in inv.items():
        if len(ns) > 1:
            ok = set(ns) <= {"s26_ctrl1", "s26_ctrl2"}
            print("  %-9s %s" % (h, "OK(控制题同素材)" if ok
                                 else "🔴 撞车 -> %s" % ns))
            if not ok:
                bad = True
    if not bad:
        print("  ✅ 无撞车")

    # ================= 每题都对应一个决策 =================
    items = [
        {"id": "s1_accept_20261015s", "dim": "接受度",
         "title": "如果这是最终效果，你能接受吗？",
         "desc": "一段男声，30 秒，是我们现在的版本。\n\n"
                 "**就按「实际工作能不能用」判断**，不用管技术。",
         "ref": {"label": "▶ 点这个播放", "file": "s26_cur_m.mp3",
                 "note": "我们现在的版本。"},
         "note": "🔴 拿到答案后的动作：\n"
                 "  选①② → **换引擎**（所有后处理能调的维度已全部排除：\n"
                 "          频谱/语速/EQ/底噪实验证伪/MP3编码/停顿/情绪/位置/字间）\n"
                 "  选③   → **带着缺陷继续做，别再优化音质**，方向整个换掉",
         "options": [
             {"key": "A", "file": None, "label": "① 完全不能接受，一听就是合成的"},
             {"key": "B", "file": None, "label": "② 能接受，但会明显听出跟真人不同"},
             {"key": "C", "file": None, "label": "③ 能接受，不影响使用"},
             {"key": "D", "file": None, "label": "④ 说不上来"}]},

        {"id": "s2_what_matters_20261015s", "dim": "优先级",
         "title": "假设只能改好一件事，你最想要哪个？",
         "desc": "**只选一个**，按对你工作的实际影响选。",
         "ref": {"label": "▶ 点这个播放", "file": "s26_cur_m.mp3",
                 "note": "我们现在的版本。"},
         "note": "🔴 拿到答案后的动作：**决定下一阶段的工作顺序**。\n"
                 "前三轮我们一直按「机器指标」猜方向，而机器指标已全部排除干净。\n"
                 "⚠️ 如果多数人选的不是音质（⑤生成快 / ⑥能直接用），\n"
                 "   **就别再优化音质了**。",
         "options": [
             {"key": "A", "file": None, "label": "① 听起来更自然（去掉电子感）"},
             {"key": "B", "file": None, "label": "② 语速更合适"},
             {"key": "C", "file": None, "label": "③ 断句停顿更自然"},
             {"key": "D", "file": None, "label": "④ 情绪更像真人"},
             {"key": "E", "file": None, "label": "⑤ 生成更快"},
             {"key": "F", "file": None, "label": "⑥ 只要能直接用就行"}]},
    ]

    # ---- 引擎对照：素材就绪才出，否则不出（不出废题）----
    if "s26_eng_omnivoice_m" in made:
        items.append(
            {"id": "s3_engine_ab_20261015s", "dim": "引擎对照",
             "title": "这两段，哪一段更能发给客户？",
             "desc": "两段是**不同引擎**生成的，同一段文案，各 30 秒。\n\n"
                     "就按「能不能直接发」判断。",
             "ref": None,
             "note": "🔴 拿到答案后的动作：**决定男声要不要换引擎**。\n"
                     "第1段=现役 VoxCPM2，第2段=OmniVoice，两个都在跑、都能上线。\n"
                     "这不是让员工做技术选型，只是问「哪个产出更可用」。",
             "options": [
                 {"key": "A", "file": "s26_cur_m.mp3",
                  "label": "第 1 段（现役引擎）"},
                 {"key": "B", "file": "s26_eng_omnivoice_m.mp3",
                  "label": "第 2 段（OmniVoice）"},
                 {"key": "C", "file": None, "label": "两段都不能发"}]})
    else:
        print("  (OmniVoice 素材未就绪 → 跳过引擎对照题，不出空题)")

    items.append(
        {"id": "s4_send_baseline_20261015s", "dim": "可发送性",
         "title": "这 30 秒，哪一段可以直接发给客户？",
         "desc": "两段都是男声，各 30 秒，同一段文案。\n\n"
                 "按「能不能直接发给客户」判断。",
         "ref": None,
         "note": "🔴 这题是**长期基线锚点**，不是重复提问。\n"
                 "连续三轮 13 张票全选 EL；哪天出现「我们也能发」，才算真有进展。\n"
                 "→ 有了它，每次改动的效果才有参照物。",
         "options": [
             {"key": "A", "file": "s26_cur_m.mp3", "label": "第 1 段（我们的）"},
             {"key": "B", "file": "s26_el_m.mp3", "label": "第 2 段（EL）"},
             {"key": "C", "file": None, "label": "都不行"}]})

    items.append(
        {"id": "ctrl_pace_20261015s", "dim": "答题有效性校验",
         "title": "【最后一题】这两段，哪一段念得更快？",
         "desc": "**这道题必须选一个**（选错也没关系）。\n\n"
                 "同一段音频放了两遍，各 30 秒。听 5 秒就够。",
         "ref": None,
         "note": "隐藏控制题：正解=C 一样快。选 A/B 说明没在听，整份作废。",
         "options": [
             {"key": "A", "file": "s26_ctrl1.mp3", "label": "第 1 段念得更快"},
             {"key": "B", "file": "s26_ctrl2.mp3", "label": "第 2 段念得更快"},
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
        shutil.copyfile(p, p + ".bak-before-s")
        print("\n已备份旧题库 ->", p + ".bak-before-s")
    json.dump({"items": items}, open(p, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("题库已写入 %s  共 %d 题" % (p, len(items)))
    if bad or empty:
        print("🔴 存在拦截项，先别让员工答题")
    print("\n发布后确认：curl -s http://服务器:8000/api/fb/where")


if __name__ == "__main__":
    main()