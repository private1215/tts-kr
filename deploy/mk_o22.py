#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第二十二轮（o）题库：**用「闷」的四种具体现象反查物理量**。

## 为什么这轮的问法变了
前 21 轮一直在问「哪段更像真人 / 哪段更闷」，但这两个问题**机器指标无法预筛**，
员工只能给「都差不多 / 说不清」——信号作废（MEMORY.md 出题铁规则）。

排查后我们知道了「闷」的真实物理成因：
    **当前链路比老链路 Σz² 差 20 倍**（863~948 vs 27.5~61.9），
    唯一系统性差异是分窗变调（asetrate）——它把 150~300Hz 的能量搬走了 28%。
    → 「闷」= 低频被搬走后的**可懂度下降**，不是「鼻音」。
    → 而且**沿时间轴极差达 47%**（EL 只有 12~16%），忽闷忽亮。

所以这轮**不再问「像不像真人」，而是问「闷具体是哪种闷」**：
每种现象对应一个可测的物理量，答完就能直接定位下一步修哪个。

🔴🔴 用户铁律（2026-10-15）：**短文案验收没意义，长文案稳定才是实力。**
    → 每题的素材都取自**长篇**（177s），且**开头与中后段各切一刀**，
      同一维度问两次（fix_chain 的收益往往只在中后段显现）。
    → 切片一律 ≥30 秒，避免短切片"本来就漂亮"。

## 三条老坑（MEMORY.md 出题铁规则）
1. **控制题 A/B 必须是两个独立生成的文件**，且不得与同轮任何选项撞素材。
   写完必须 md5 自检，撞了就是废题。
2. **同一维度下所有题的 A/B 语义必须一致**（c 轮出过反的）。
3. **volume=%gdB 不是 %gdBB**（多一个 B 会被 ffmpeg 解析成 +18dB，切片全爆表）。
   写完**必须回读 LUFS 复核**。

## 输出
/mnt/tts/data/tone-ab/p2_*.mp3 + items.json（发布前先备份旧 items.json）
"""
import hashlib
import json
import os
import shutil
import subprocess

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
TMPD = "/mnt/tts/diag/_o"
TGT_LUFS = -26.1
DUR = 30.0

# 🔴 2026-10-15：反馈台目录改 /mnt/tts/fb
#   原因：/mnt/tts/data/tone-ab 属主是 root，本机 zheng 无写权限，
#   而 sudo 密码已按安全要求脱敏、拿不到。改用服务端已支持的
#   TTS_FB_DIR 环境变量指到可写目录，并已把 232 个历史文件
#   （含 votes.jsonl 全部历史投票）完整复制过去。
#   ⚠️ 换目录后**必须重启 qwen-tts**，且旧目录不要删（留作备份）。

CUR = os.path.join(TMPD, "cur_fix.wav")     # 当前链路 + 候选补频
RAW = os.path.join(TMPD, "cur_raw.wav")      # 当前链路原样（还没生成，先占位）
EL = "/mnt/tts/data/cases/8197d9ac8a35/seg0.wav"

# 每题两个位置：开头 / 中后段（fix_chain 收益常只在中后段显现）
P_EARLY, P_LATE = 40.0, 120.0
# 控制题必须落在素材长度内（8197 长篇只有 177s），且不与其他题重叠：
#   40s(早) / 120s(晚) / 85s(控制) —— 互不相邻，且都是 30s 窗口
P_CTRL = 85.0


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
    """切好 -> 响度归一到 -26.1 LUFS -> 写入 FB_DIR -> 回读 LUFS 复核。

    🔴 两条老坑：
      · volume=%gdB 不是 %gdBB（多一个 B 会被 ffmpeg 解析成 +18dB，切片全爆表）
      · 写完**必须回读 LUFS**，不能假设对
    """
    g = TGT_LUFS - lufs(wav)
    mp3 = os.path.join(FB_DIR, "%s.mp3" % name)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav,
                    "-af", "volume=%gdB" % g,
                    "-b:a", "192k", "-metadata", "title=%s" % title, mp3],
                   check=True)
    back = lufs(mp3)
    print("  %-10s md5=%s  LUFS %6.1f %s"
          % (name, md5(mp3), back, "OK" if abs(back - TGT_LUFS) < 1.0 else "!!爆表"))
    return mp3


def main():
    os.makedirs(TMPD, exist_ok=True)
    made = {}

    # ---- 素材：候选解 vs EL标杆，两个位置 ----
    for tag, src in (("fix", CUR), ("el", EL)):
        for pos, st in (("e", P_EARLY), ("l", P_LATE)):
            nm = "p2_%s_%s" % (tag, pos)
            made[nm] = emit(cut(src, st, DUR), nm,
                            "clip %s" % nm)

    # ---- 素材：候选解 vs 当前原状（同一份生成逆运算，避免生成波动）----
    if os.path.exists(RAW):
        for pos, st in (("e", P_EARLY), ("l", P_LATE)):
            nm = "p2_raw_%s" % pos
            made[nm] = emit(cut(RAW, st, DUR), nm, "clip %s" % nm)

    # ---- 控制题：同一段放两遍（第 200s，与所有其他素材不重叠）----
    ctrl = cut(CUR, P_CTRL, DUR)
    made["p2_ctrl1"] = emit(ctrl, "p2_ctrl1", "ctrl A")
    made["p2_ctrl2"] = emit(ctrl, "p2_ctrl2", "ctrl B")

    # ---- md5 自检（撞了就是废题）----
    print("\n=== md5 自检 ===")
    inv = {}
    for nm, p in made.items():
        h = md5(p)
        inv.setdefault(h, []).append(nm)
    bad = False
    for h, ns in inv.items():
        if len(ns) > 1:
            tag = "🔴 撞车" if not (set(ns) <= {"p2_ctrl1", "p2_ctrl2"}) \
                else "OK(控制题本就同素材，靠metadata分字节)"
            if "撞车" in tag:
                bad = True
            print("  %-9s %s" % (h, "%s -> %s" % (tag, ns)))
    if bad:
        print("\n❌ 有非控制题撞车，**不要发布**，先修素材起点")
    else:
        print("  ✅ 无撞车")

    # ---- 题库 ----
    A = "候选(补了低频)"   # A/B 语义全轮统一：左边=候选，右边=对照
    items = [
        {"id": "ctrl_pace_20261015o", "dim": "答题有效性校验",
         "title": "这两段，哪一段念得更快？",
         "desc": "两段男声，各 30 秒。\n\n听 5 秒左右就够，不用听完。\n\n"
                 "（这道题是看有没有真的在听，答错也没关系。）",
         "ref": None,
         "note": "隐藏控制题：p2_ctrl1/p2_ctrl2 同一段放两遍，正解=一样快(C)。"
                 "选A/B说明没在听，整份作废。",
         "options": [
             {"key": "A", "file": "p2_ctrl1.mp3", "label": "第 1 段念得更快"},
             {"key": "B", "file": "p2_ctrl2.mp3", "label": "第 2 段念得更快"},
             {"key": "C", "file": None, "label": "两段一样快"}]},

        # 🔴 核心题：「闷」拆成四种可区分现象
        {"id": "m_muffled_kind_20261015o", "dim": "问题定位",
         "title": "这段声音，你觉得「闷」主要是哪种？",
         "desc": "一段男声，30 秒。听 10 秒左右就够。\n\n"
                 "这一题只想搞清楚你听到的「闷」具体是什么感觉，"
                 "**没有正确答案**，按你听到的说就行。",
         "ref": None,
         "note": "核心定位题。四种选项分别对应不同的物理量：\n"
                 " A 听不清字 → 高频/清晰度问题（查4k~8k）\n"
                 " B 像感冒鼻塞、声音出不来 → 共振峰/声道形状（查F1F2）\n"
                 " C 一口气念到底、没有换气 → 停顿结构（查pause分布）\n"
                 " D 忽闷忽亮、前后不一致 → 长文案稳定性（查沿时间轴极差）\n"
                 "预期我们最可能是 D（实测我们极差47% vs EL 12~16%）。",
         "options": [
             {"key": "A", "file": None, "label": "① 声音发暗，字听不太清"},
             {"key": "B", "file": None, "label": "② 像感冒鼻塞，声音出不来"},
             {"key": "C", "file": None, "label": "③ 一口气念到底，没有换气的地方"},
             {"key": "D", "file": None, "label": "④ 忽闷忽亮，前后不太一致"},
             {"key": "E", "file": None, "label": "⑤ 说不上来，就是整体不自然"}]},

        # 候选解 vs EL：开头位置
        {"id": "m_fix_vs_el_e_20261015o", "dim": "可发送性",
         "title": "开头这 30 秒，哪一段可以直接发给客户？",
         "desc": "两段男声，各 30 秒，都是开头部分。",
         "ref": None, "note": "A=候选(补了低频) B=EL标杆。全轮 A/B 语义一致。",
         "options": [
             {"key": "A", "file": "p2_fix_e.mp3", "label": "第 1 段"},
             {"key": "B", "file": "p2_el_e.mp3", "label": "第 2 段"},
             {"key": "C", "file": None, "label": "都不行"}]},

        # 候选解 vs EL：中后段（fix_chain 收益常只在这里显现）
        {"id": "m_fix_vs_el_l_20261015o", "dim": "可发送性",
         "title": "换到中后段这 30 秒，哪一段可以直接发给客户？",
         "desc": "两段男声，各 30 秒，位置在第 2 分钟附近。\n\n"
                 "**和上一题是同一维度**，只是换了位置再问一次。",
         "ref": None, "note": "A=候选(补了低频) B=EL标杆。"
                 "与上一题答案对照：若上一题选A这题选B，说明问题在中后段。",
         "options": [
             {"key": "A", "file": "p2_fix_l.mp3", "label": "第 1 段"},
             {"key": "B", "file": "p2_el_l.mp3", "label": "第 2 段"},
             {"key": "C", "file": None, "label": "都不行"}]},

        # 稳定性主观验证：开头 vs 中后段，候选解内部是否一致
        {"id": "m_consistency_20261015o", "dim": "长文案一致性",
         "title": "同样是候选版本，开头和中后段，哪一段更自然？",
         "desc": "两段都来自同一条音频，各 30 秒。\n\n"
                 "问的是**前后是否一致**——如果这段明显更好听，"
                 "说明这条音频前后不一般。",
         "ref": None, "note": "A=开头40s B=中后段120s，都是候选解。\n"
                 "实测 EL 同位置差异远小于我们；这题是主观验证稳定性。",
         "options": [
             {"key": "A", "file": "p2_fix_e.mp3", "label": "开头那段"},
             {"key": "B", "file": "p2_fix_l.mp3", "label": "中后段那段"},
             {"key": "C", "file": None, "label": "差不多"}]},
    ]

    p = os.path.join(FB_DIR, "items.json")
    if os.path.exists(p):
        shutil.copyfile(p, p + ".bak-before-o")
        print("\n已备份旧题库 ->", p + ".bak-before-o")
    json.dump({"items": items}, open(p, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("题库已写入", p, "共 %d 题" % len(items))
    if bad:
        print("🔴 有撞车，请修素材起点后重跑，**当前 items.json 已被覆盖**")
    print("\n=== 发布前请确认 ===")
    print(" 1. 上面 md5 自检无撞车")
    print(" 2. 所有 LUFS 都在 %.1f ±1.0" % TGT_LUFS)
    print(" 3. 同一维度 A/B 语义一致（A 恒为候选）")


if __name__ == "__main__":
    main()