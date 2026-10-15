#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第二十三轮（p）题库：**把所有不确定项都交给员工判断**。

## 为什么这样出题（2026-10-15 用户指令）
> **「下午有3个员工来做反馈题，下午的更新需要频繁一些，
>   不确定的数据因素都做到反馈里给员工去反馈，然后对应调整。」**

下午 3 人同场答题，要满足：
1. **每题只问一件事**——3 个人独立作答，互不影响，不能有"上一题依赖下一题"
2. **所有不确定项全部出题**——机器测不准的（听感、真实感、可发送性）就交给耳朵
3. **选项说人话 + 给"说不清"出口**——但不能只留一个笼统出口
4. **不强制听完整段**——凭感觉即可，播放秒数被动记录

## 这一轮要问的「不确定项」清单（全部来自 10-15 排查）
| # | 不确定项 | 机器能测吗 | 对应题 |
|---|---|---|---|
| U1 | 「闷」到底是哪种 | ❌ 证伪了（改频段没用） | 第2题 |
| U2 | 候选补频能否被听出 | ⚠️ 只有机器分 | 第3/4题 |
| U3 | 前后是否一致（长文案稳定性） | ⚠️ 只有极差% | 第5题 |
| U4 | **语速是否真的偏慢** | ✅ 测出 4.07 vs EL 8.37 字/秒 | 第6题 |
| U5 | 到底哪里最影响「像真人」 | ❌ 一直是猜测 | 第7题 |
| U6 | **能否直接发客户**（唯一真正的验收） | ❌ | 第8/9题 |
| U7 | 有没有换气/停顿问题 | ⚠️ 只测过总静音量 | 第10题 |

🔴 U4 是本轮最重要的补充：**当前链路 4.07 字/秒，EL 锚点 8.37，偏慢 51%**。
这个必须让耳朵确认——机器算得出字/秒，但「听着慢不慢」只有人能判断。

## 三条老坑（MEMORY.md 出题铁规则，违反就废题）
1. **md5 撞车自检**：控制题 A/B 靠 `-metadata title=` 分字节，其余撞了就是废题
2. **LUFS 回读**：`volume=%gdB` 不是 `%gdBB`（多一个 B = +18dB，全爆表）
3. **题目完整性**：既无选项音频也无 `ref` 的题是**纯文字废题**，员工无法作答

## 输出
/mnt/tts/fb/p3_*.mp3 + items.json（旧题库备份为 items.json.bak-before-p）
"""
import hashlib
import json
import os
import shutil
import subprocess

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
TMPD = "/mnt/tts/diag/_p3"
TGT_LUFS = -26.1
DUR = 30.0

CUR = "/mnt/tts/diag/_o/cur_fix.wav"     # 当前链路 + 候选补频
RAW = "/mnt/tts/diag/_o/cur_raw.mp3"     # 当前链路原状（线上真实状态）
EL_M = "/mnt/tts/data/cases/8197d9ac8a35/seg0.wav"   # EL 男声标杆
EL_F = "/mnt/tts/data/cases/00b6a45ad6aa/seg0.wav"   # EL 女声标杆（若存在）

# 🔴 起点必须互不重叠（30 秒窗口，起点间隔 ≥30s），且都在素材时长内
#   8197 男声长篇 177s；40/75/110/145 都安全；145+30=175 < 177
P_E, P_M, P_L = 40.0, 75.0, 110.0
P_CTRL = 145.0


def lufs(p):
    r = subprocess.run(["ffmpeg", "-i", p, "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    v = [l for l in r.stderr.splitlines() if "I:" in l and "LUFS" in l]
    return float(v[-1].split("I:")[1].split("LUFS")[0].strip()) if v else None


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()[:8]


def cut(src, start, dur):
    o = os.path.join(TMPD, "_c_%.0f.wav" % start)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "%.1f" % start,
                    "-i", src, "-t", "%.1f" % dur, o], check=True)
    return o


def emit(wav, name, title):
    g = TGT_LUFS - lufs(wav)
    mp3 = os.path.join(FB_DIR, "%s.mp3" % name)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav,
                    "-af", "volume=%gdB" % g,      # 🔴 dB 不是 %gdBB
                    "-b:a", "192k", "-metadata", "title=%s" % title, mp3],
                   check=True)
    back = lufs(mp3)
    ok = "OK" if abs(back - TGT_LUFS) < 1.0 else "!!爆表"
    print("  %-12s md5=%s  LUFS %6.1f %s" % (name, md5(mp3), back, ok))
    return mp3


def main():
    os.makedirs(TMPD, exist_ok=True)
    made = {}

    # ---- 素材表：raw(线上原状) / fix(候选补频) / el(标杆)，各三个位置 ----
    SRC = {"raw": RAW, "fix": CUR, "el": EL_M}
    for tag, src in SRC.items():
        if not os.path.exists(src):
            print("🔴 缺素材 %s" % src)
            continue
        for pos, st in (("e", P_E), ("m", P_M), ("l", P_L)):
            nm = "p3_%s_%s" % (tag, pos)
            made[nm] = emit(cut(src, st, DUR), nm, "clip %s" % nm)

    # ---- 控制题：同一段放两遍（起点 145s，与上面不重叠）----
    ctrl = cut(CUR, P_CTRL, DUR)
    made["p3_ctrl1"] = emit(ctrl, "p3_ctrl1", "ctrl A")
    made["p3_ctrl2"] = emit(ctrl, "p3_ctrl2", "ctrl B")

    # ---- md5 自检 ----
    print("\n=== md5 自检 ===")
    inv = {}
    for nm, p in made.items():
        inv.setdefault(md5(p), []).append(nm)
    bad = False
    for h, ns in inv.items():
        if len(ns) > 1:
            allowed = set(ns) <= {"p3_ctrl1", "p3_ctrl2"}
            print("  %-9s %s" % (h, "OK(控制题同素材)" if allowed
                                 else "🔴 撞车 -> %s" % ns))
            if not allowed:
                bad = True
    if not bad:
        print("  ✅ 无撞车")

    # ============================================================
    # 题目：每题只问一件事；A/B 语义全轮一致（A 恒为「我们的」）
    # ============================================================
    items = [
        # --- 0. 控制题（隐藏）---
        {"id": "ctrl_pace_20261015p", "dim": "答题有效性校验",
         "title": "这两段，哪一段念得更快？",
         "desc": "同一段音频放了两遍，各 30 秒。\n\n听 5 秒左右就够。\n\n"
                 "（这题是看有没有在听，答错没关系。）",
         "ref": None,
         "note": "隐藏控制题：p3_ctrl1/p3_ctrl2 同素材，正解=C 一样快。"
                 "选A/B说明没在听，这一整份作废。",
         "options": [
             {"key": "A", "file": "p3_ctrl1.mp3", "label": "第 1 段念得更快"},
             {"key": "B", "file": "p3_ctrl2.mp3", "label": "第 2 段念得更快"},
             {"key": "C", "file": None, "label": "两段一样快"}]},

        # --- 1. U1：闷到底是哪种（核心）---
        {"id": "u1_muffled_kind_20261015p", "dim": "问题定位",
         "title": "这段声音，你觉得「闷」主要是哪种？",
         "desc": "一段男声，30 秒，是我们现在的版本。\n\n"
                 "听完选一个就行，**没有正确答案**，按你听到的说。",
         "ref": {"label": "▶ 点这个播放", "file": "p3_raw_m.mp3",
                 "note": "就是我们现在的版本，不是 EL。"},
         "note": "U1。机器已证伪「闷=低频过多」（该频段从+3.96σ压到−3.08σ，员工照样说闷）。\n"
                 "选项对应不同物理量：\n"
                 " ① 字听不清 → 高频不足（查4k~8k，短片实测只有EL的0.26~0.54）\n"
                 " ② 像感冒鼻塞 → 共振峰/声道形状（查F1F2）\n"
                 " ③ 一口气没换气 → 停顿结构\n"
                 " ④ 忽闷忽亮 → 长文案稳定性（查沿时间轴极差）\n"
                 "预期 ④ 概率最高（我们极差47% vs EL 12~16%）。",
         "options": [
             {"key": "A", "file": None, "label": "① 声音发暗，字听不太清"},
             {"key": "B", "file": None, "label": "② 像感冒鼻塞，声音出不来"},
             {"key": "C", "file": None, "label": "③ 一口气念到底，中间没换气"},
             {"key": "D", "file": None, "label": "④ 忽闷忽亮，前后不太一致"},
             {"key": "E", "file": None, "label": "⑤ 以上都不像，就是整体不自然"}]},

        # --- 2. U4：语速（机器测出偏慢 51%，要耳朵确认）---
        {"id": "u4_speed_20261015p", "dim": "语速",
         "title": "这两段，哪一段念得快一些？",
         "desc": "两段都是男声，各 30 秒。\n\n"
                 "第 1 段是我们的版本，第 2 段是 ElevenLabs 的。\n\n"
                 "就判断哪个念得快，不用管别的。",
         "ref": None,
         "note": "U4。机器测出：我们的 4.07 字/秒 vs EL 8.37，偏慢 51%（老链路 7.88 正常）。\n"
                 "**A/B 语义：A=我们，B=EL，全轮一致。**\n"
                 "若选 A（我们更快）→ 与机器结论矛盾，说明机器口径有问题（查字/秒统计）。\n"
                 "若选 B 或 C（我们更慢/差不多）→ 证实语速问题，去查 fix_chain 的 --target-cps。",
         "options": [
             {"key": "A", "file": "p3_raw_e.mp3", "label": "第 1 段（我们的）念得更快"},
             {"key": "B", "file": "p3_el_e.mp3", "label": "第 2 段（EL）念得更快"},
             {"key": "C", "file": None, "label": "差不多"}]},

        # --- 3. U2：候选补频能否被听出（开头位置）---
        {"id": "u2_fix_vs_raw_e_20261015p", "dim": "改动可听度",
         "title": "开头这 30 秒，哪一段听着更舒服？",
         "desc": "两段都是男声，各 30 秒，都取自开头。\n\n"
                 "第 1 段是我们调整后的版本，第 2 段是调整前。",
         "ref": None,
         "note": "U2。机器上候选补频让 Σz² 683→108，但**这不代表能听出来**。\n"
                 "**A/B 语义：A=调整后，B=调整前，全轮一致。**\n"
                 "选 C 说明耳朵听不出 → 那就该改指标口径而不是继续调 EQ。",
         "options": [
             {"key": "A", "file": "p3_fix_e.mp3", "label": "第 1 段（调整后）更舒服"},
             {"key": "B", "file": "p3_raw_e.mp3", "label": "第 2 段（调整前）更舒服"},
             {"key": "C", "file": None, "label": "差不多，听不出差别"}]},

        # --- 4. U2 换位置（中后段：fix_chain 收益常只在这里显现）---
        {"id": "u2_fix_vs_raw_l_20261015p", "dim": "改动可听度",
         "title": "换到中后段这 30 秒，哪一段听着更舒服？",
         "desc": "两段都是男声，各 30 秒，位置在第 2 分钟附近。\n\n"
                 "**和上一题是同一件事**，只是换了位置再问一次。",
         "ref": None,
         "note": "U2 的第二个位置。\n"
                 "**判读**：与上一题对照——若上一题选 A 这题选 B，说明改善只在前段；\n"
                 "两题都选 A → 候选补频真的有效，可考虑上线。",
         "options": [
             {"key": "A", "file": "p3_fix_l.mp3", "label": "第 1 段（调整后）更舒服"},
             {"key": "B", "file": "p3_raw_l.mp3", "label": "第 2 段（调整前）更舒服"},
             {"key": "C", "file": None, "label": "差不多，听不出差别"}]},

        # --- 5. U3：前后一致性（长文案稳定性）---
        {"id": "u3_consistency_20261015p", "dim": "长文案一致性",
         "title": "同样是我们的版本，开头和中后段，哪一段更自然？",
         "desc": "两段都来自同一条音频，各 30 秒。\n\n"
                 "如果其中一段明显更好听，说明这条音频前后不一般。",
         "ref": None,
         "note": "U3。机器上我们沿时间轴极差 47%（EL 只有 12~16%），这题是主观验证。\n"
                 "A=开头 40s / B=中后段 110s。选哪段更好，就说明**问题集中在另一段**。",
         "options": [
             {"key": "A", "file": "p3_raw_e.mp3", "label": "开头那段"},
             {"key": "B", "file": "p3_raw_l.mp3", "label": "中后段那段"},
             {"key": "C", "file": None, "label": "差不多"}]},

        # --- 6. U6：能否直接发客户（开头）---
        {"id": "u6_send_e_20261015p", "dim": "可发送性",
         "title": "开头这 30 秒，哪一段可以直接发给客户？",
         "desc": "两段都是男声，各 30 秒。\n\n"
                 "就按「能不能直接发给客户」判断，不用细究原因。",
         "ref": None,
         "note": "U6。**这是唯一真正的验收标准**。\n"
                 "A=我们 B=EL。若选 C（都不行）→ 说明当前版本还不能上线。",
         "options": [
             {"key": "A", "file": "p3_raw_e.mp3", "label": "第 1 段（我们的）"},
             {"key": "B", "file": "p3_el_e.mp3", "label": "第 2 段（EL）"},
             {"key": "C", "file": None, "label": "都不行"}]},

        # --- 7. U6 换位置（中后段）---
        {"id": "u6_send_l_20261015p", "dim": "可发送性",
         "title": "换到中后段这 30 秒，哪一段可以直接发给客户？",
         "desc": "两段都是男声，各 30 秒，位置在第 2 分钟附近。",
         "ref": None,
         "note": "U6 的第二个位置。与上一题对照可看出问题是否集中在中后段。",
         "options": [
             {"key": "A", "file": "p3_raw_l.mp3", "label": "第 1 段（我们的）"},
             {"key": "B", "file": "p3_el_l.mp3", "label": "第 2 段（EL）"},
             {"key": "C", "file": None, "label": "都不行"}]},

        # --- 8. U7：换气/停顿 ---
        {"id": "u7_breath_20261015p", "dim": "停顿与换气",
         "title": "这段听着累不累？主要是哪种情况？",
         "desc": "一段男声，30 秒，是我们现在的版本。\n\n按你听到的说。",
         "ref": {"label": "▶ 点这个播放", "file": "p3_raw_m.mp3",
                 "note": "我们的版本。"},
         "note": "U7。之前只测过「总静音量」，从没测过停顿**分布**，这是空白。\n"
                 " ① → 一口气到底，该加停顿（查 pause 分布）\n"
                 " ② → 停顿太多太碎，该合并\n"
                 " ③ → 停顿位置不对（该断的地方没断）",
         "options": [
             {"key": "A", "file": None, "label": "① 听着一口气到底，中间没换气的地方"},
             {"key": "B", "file": None, "label": "② 停顿太多太碎，断得很别扭"},
             {"key": "C", "file": None, "label": "③ 停顿位置不对，该断的地方没断"},
             {"key": "D", "file": None, "label": "④ 停顿这块没什么问题"}]},

        # --- 9. U5：最影响真实感的是什么（单选，多维归因）---
        {"id": "u5_top_issue_20261015p", "dim": "优先级排序",
         "title": "如果只能改一处来让它更像真人，你最想改哪里？",
         "desc": "先听上面的音频，再选。\n\n"
                 "**只选一个**——如果只能改一处，你会改哪处？",
         "ref": {"label": "▶ 再听一次", "file": "p3_raw_m.mp3",
                 "note": "还是我们现在的版本。"},
         "note": "U5。之前一直靠猜「闷」到底是什么，没有优先级依据。\n"
                 "这张票直接给出「先修哪个」的顺序，比任何单维指标都管用。\n"
                 "⚠️ 三个人都投的选项 = 最高优先级。",
         "options": [
             {"key": "A", "file": None, "label": "① 声音太闷/发暗，听不清字"},
             {"key": "B", "file": None, "label": "② 语速不对（太快或太慢）"},
             {"key": "C", "file": None, "label": "③ 断句停顿不自然"},
             {"key": "D", "file": None, "label": "④ 前后忽明忽暗，不一致"},
             {"key": "E", "file": None, "label": "⑤ 情绪太平，像在念稿"},
             {"key": "F", "file": None, "label": "⑥ 说不上来，就是整体不像"}]},
    ]

    # ---- 题目完整性（纯文字题 = 废题）----
    print("\n=== 题目完整性 ===")
    empty = []
    for it in items:
        hf = any(o.get("file") for o in it.get("options") or [])
        hr = bool(it.get("ref") and it["ref"].get("file"))
        if not (hf or hr):
            empty.append(it["id"])
        print("  %-30s 选项音频=%s 参考音频=%s"
              % (it["id"], "有" if hf else "无", "有" if hr else "无"))
    print("  ✅ 每题都有音频" if not empty
          else "  🔴 废题（无音频）: %s" % empty)

    p = os.path.join(FB_DIR, "items.json")
    if os.path.exists(p):
        shutil.copyfile(p, p + ".bak-before-p")
        print("\n已备份旧题库 ->", p + ".bak-before-p")
    json.dump({"items": items}, open(p, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("题库已写入 %s  共 %d 题" % (p, len(items)))
    if bad or empty:
        print("🔴 存在拦截项，先别让员工答题")
    print("\n=== 发布后请打这个端点确认 ===")
    print("  curl -s http://服务器:8000/api/fb/where | grep -o 'missing.*'")


if __name__ == "__main__":
    main()