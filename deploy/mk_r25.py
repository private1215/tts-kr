#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第二十五轮（r）题库：**把「电音感」问透**（第24轮 2/2 一致的新焦点）

## 为什么这么出题
第 24 轮拿到 **2/2 一致**：像真人录的还是电子合成 → **③ 明显像电子合成的**，
且优先级题 **2/2 把它排第一**，고구마 **连续两轮**写「电音严重」。

但**第 24 轮把「电音感」定位到具体成因的尝试全部失败**：
| 候选 | 实测结果 |
|---|---|
| 合成底噪 | ❌ 受控实验证伪（静音段只降 1.8dB，两份最静段都是 −240dB 数字真零） |
| MP3 编码 | ❌ 我们 44.1k/192k，EL 44.1k/128k，**同规格** |
| 静态频谱 | ❌ 9标量+HNR+质心+F1F2 全在 EL 分布内 |
| 语速 | ❌ 8.36 vs 8.37 |
| 候选补频 EQ | ❌ 开头位置三人各选一个，听不出 |

→ 剩下最可能是**引擎本身的合成特质**，属引擎选型问题，不是后处理能修的。

🔴 **但"像电子合成"太笼统，无法定位。** 这一轮的核心任务就是把它拆细：
问「**具体像什么**」，而不是问「像不像真人」。

## 本轮设计（按记忆里已验证有效的出题原则）
✅ **宁可给具体选项，也不要兜底选项**（第23轮 3/3 选兜底 = 信号作废；
   第24轮去掉兜底立刻拿到 2/2 具体答案 → 原则已被验证）

素材全部取自 **496 秒真长文**（`ab-rt-base`，含底噪的现役版本）。
"""
import hashlib
import json
import os
import shutil
import subprocess

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
TMPD = "/mnt/tts/diag/_r25"
TGT_LUFS = -26.1
DUR = 30.0

CUR = "/mnt/tts/output/ab-rt-base/ab-rt-base-1.mp3"     # 现役（496s 真长文）
EL_S = "/mnt/tts/data/cases/6cdd25bc75f2/merged.wav"   # EL 标杆 509s

# 起点互不重叠，全在 496/509s 内
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
        print("  %-10s 🔴 读不出 LUFS" % name)
        return None
    print("  %-10s md5=%s  LUFS %6.1f %s"
          % (name, md5(mp3), back,
             "OK" if abs(back - TGT_LUFS) < 1.0 else "!!爆表"))
    return mp3


def main():
    os.makedirs(TMPD, exist_ok=True)
    made = {}
    for tag, src in (("cur", CUR), ("el", EL_S)):
        for pos, st in (("e", P_E), ("m", P_M), ("l", P_L)):
            nm = "r25_%s_%s" % (tag, pos)
            made[nm] = emit(cut(src, st, DUR), nm, "clip %s" % nm)

    ctrl = cut(CUR, 150.0, DUR)
    made["r25_ctrl1"] = emit(ctrl, "r25_ctrl1", "ctrl A")
    made["r25_ctrl2"] = emit(ctrl, "r25_ctrl2", "ctrl B")

    print("\n=== md5 自检 ===")
    inv = {}
    for nm, p in made.items():
        inv.setdefault(md5(p), []).append(nm)
    bad = False
    for h, ns in inv.items():
        if len(ns) > 1:
            ok = set(ns) <= {"r25_ctrl1", "r25_ctrl2"}
            print("  %-9s %s" % (h, "OK(控制题同素材)" if ok
                                 else "🔴 撞车 -> %s" % ns))
            if not ok:
                bad = True
    if not bad:
        print("  ✅ 无撞车")

    items = [
        # ===== 核心：把「电音感」拆成具体形态 =====
        {"id": "r1_electronic_shape_20261015r", "dim": "电音感（形态）",
         "title": "这段的「电子感」具体像什么？",
         "desc": "一段男声，30 秒，是我们现在的版本。\n\n"
                 "**如果听着像电子合成的，最像下面哪一种？**\n"
                 "如果完全听不出电子感，就选最后一项。",
         "ref": {"label": "▶ 点这个播放", "file": "r25_cur_m.mp3",
                 "note": "我们现在的版本。"},
         "note": "🔴 本轮核心题。第24轮 2/2 说「像电子合成」但太笼统。\n"
                 "**已排除**：合成底噪（受控实验证伪）、MP3编码（同规格）、\n"
                 "静态频谱、语速、EQ。所以要问具体形态才能定位到引擎层。\n"
                 "预期：如果多数人选①电流/电子蜂鸣 → 查高频与谐波结构；\n"
                 "      ②金属/机器人 → 查共振峰与音色；\n"
                 "      ③合成器播报 → 查韵律与情绪动态。",
         "options": [
             {"key": "A", "file": None, "label": "① 像有电流声/电子蜂鸣声"},
             {"key": "B", "file": None, "label": "② 像金属/机器人的声音"},
             {"key": "C", "file": None, "label": "③ 像合成器播报，一板一眼"},
             {"key": "D", "file": None, "label": "④ 像老式手机/调制解调器"},
             {"key": "E", "file": None, "label": "⑤ 听不出电子感，比较自然"}]},

        # ===== 与 EL 对照：电子感差多少 =====
        {"id": "r2_electronic_vs_el_20261015r", "dim": "电音感（对照）",
         "title": "这两段，哪一段更像真人录的？",
         "desc": "两段都是男声，各 30 秒，同一段文案。\n\n"
                 "第 1 段是我们的版本，第 2 段是 ElevenLabs 的。",
         "ref": None,
         "note": "A/B 语义：A=我们，B=EL。\n"
                 "与第24轮「2/2 像电子合成」对照，看**差距有多大**。",
         "options": [
             {"key": "A", "file": "r25_cur_e.mp3", "label": "第 1 段（我们的）更像真人"},
             {"key": "B", "file": "r25_el_e.mp3", "label": "第 2 段（EL）更像真人"},
             {"key": "C", "file": None, "label": "差不多"}]},

        # ===== 电子感是否随时长变化（长文案特有）=====
        {"id": "r3_electronic_position_20261015r", "dim": "电音感（位置）",
         "title": "这两段，哪一段的电子感更重？",
         "desc": "两段都来自同一条音频，各 30 秒（开头 / 中后段）。\n\n"
                 "**只问电子感**，别的不用管。",
         "ref": None,
         "note": "🔴 首版这题是纯文字题（无音频）→ 完整性校验判为废题，已修。\n"
                 "若中后段电子感更重 → 与「F0 在长篇里往上漂」同源\n"
                 "（记忆：VoxCPM2 长篇 F0 漂 +11%，EL 是平的），\n"
                 "可查是否与漂移后的分窗变调有关。",
         "options": [
             {"key": "A", "file": "r25_cur_e.mp3", "label": "开头那段电子感更重"},
             {"key": "B", "file": "r25_cur_l.mp3", "label": "中后段那段电子感更重"},
             {"key": "C", "file": None, "label": "差不多"}]},

        # ===== 念词时的电子感（元音/辅音是否发虚）=====
        {"id": "r4_electronic_words_20261015r", "dim": "电音感（清晰度）",
         "title": "听这句话的时候，每个字的发音清楚吗？",
         "desc": "一段男声，30 秒，是我们现在的版本。\n\n"
                 "重点听**字和字之间的衔接**。",
         "ref": {"label": "▶ 点这个播放", "file": "r25_cur_l.mp3",
                 "note": "我们现在的版本。"},
         "note": "🔴 排查点：神经 TTS 常见问题是**字间衔接发虚/糊在一起**\n"
                 "（声学模型过渡不够锐利）。\n"
                 "记忆里 4k~8k 沿时间轴极差 94.9%（EL 31.9%）可能与此有关。\n"
                 "若多数人选②→ 查辅音瞬态/高频细节；\n"
                 "  选③→ 说明是整体音色问题而非局部。",
         "options": [
             {"key": "A", "file": None, "label": "① 每个字都很清楚"},
             {"key": "B", "file": None, "label": "② 字和字之间有点糊、发虚"},
             {"key": "C", "file": None, "label": "③ 说不上来"}]},

        # ===== 优先级复核 =====
        {"id": "r5_priority_20261015r", "dim": "优先级排序",
         "title": "如果只能改一处，你最想先改哪里？",
         "desc": "先听完上面那段，再选。\n\n**只选一个。**",
         "ref": {"label": "▶ 再听一次", "file": "r25_cur_m.mp3",
                 "note": "我们现在的版本。"},
         "note": "第24轮此题 **2/2 选「像电子合成的」**。\n"
                 "这轮复核是否稳定；若仍第一，说明它确实是主要矛盾。",
         "options": [
             {"key": "A", "file": None, "label": "① 电子合成感"},
             {"key": "B", "file": None, "label": "② 断句停顿不自然"},
             {"key": "C", "file": None, "label": "③ 字听不清、发闷"},
             {"key": "D", "file": None, "label": "④ 情绪太平像念稿"},
             {"key": "E", "file": None, "label": "⑤ 前后忽明忽暗"},
             {"key": "F", "file": None, "label": "⑥ 都不突出，说不上来"}]},

        # ===== 最终验收 =====
        {"id": "r6_send_20261015r", "dim": "可发送性",
         "title": "这 30 秒，哪一段可以直接发给客户？",
         "desc": "两段都是男声，各 30 秒，同一段文案。\n\n"
                 "按「能不能直接发给客户」判断。",
         "ref": None,
         "note": "**唯一真正的验收标准。**\n"
                 "连续两轮 13 张票全选 EL，本题看是否仍如此。",
         "options": [
             {"key": "A", "file": "r25_cur_m.mp3", "label": "第 1 段（我们的）"},
             {"key": "B", "file": "r25_el_m.mp3", "label": "第 2 段（EL）"},
             {"key": "C", "file": None, "label": "都不行"}]},

        # ===== 控制题放最后 =====
        {"id": "ctrl_pace_20261015r", "dim": "答题有效性校验",
         "title": "【最后一题】这两段，哪一段念得更快？",
         "desc": "**这道题必须选一个**（选错没关系）。\n\n"
                 "同一段音频放了两遍，各 30 秒。听 5 秒就够。",
         "ref": None,
         "note": "隐藏控制题：正解=C 一样快。选 A/B 说明没在听，整份作废。",
         "options": [
             {"key": "A", "file": "r25_ctrl1.mp3", "label": "第 1 段念得更快"},
             {"key": "B", "file": "r25_ctrl2.mp3", "label": "第 2 段念得更快"},
             {"key": "C", "file": None, "label": "两段一样快"}]},
    ]

    print("\n=== 题目完整性 ===")
    empty = []
    for it in items:
        hf = any(o.get("file") for o in it.get("options") or [])
        hr = bool(it.get("ref") and it["ref"].get("file"))
        if not (hf or hr):
            empty.append(it["id"])
        print("  %-34s 选项=%s 参考=%s"
              % (it["id"], "有" if hf else "无", "有" if hr else "无"))
    print("  ✅ 每题都有音频" if not empty else "  🔴 废题: %s" % empty)

    p = os.path.join(FB_DIR, "items.json")
    if os.path.exists(p):
        shutil.copyfile(p, p + ".bak-before-r")
        print("\n已备份旧题库 ->", p + ".bak-before-r")
    json.dump({"items": items}, open(p, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("题库已写入 %s  共 %d 题" % (p, len(items)))
    if bad or empty:
        print("🔴 存在拦截项，先别让员工答题")
    print("\n发布后确认：curl -s http://服务器:8000/api/fb/where")


if __name__ == "__main__":
    main()