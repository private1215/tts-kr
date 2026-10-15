#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第二十四轮（q）题库：**按第 23 轮反馈的结论出题**。

## 本轮问法为什么变
第 23 轮拿到 3/4 条强信号：
- **u1「闷」是哪种 → 3/3 选兜底选项「以上都不像，整体不自然」**
  → 证明「闷」不是任何单一维度，**再问「哪种闷」是无效问法**
- **u5 优先级 ③断句停顿不自然 2/3 + u7 ②停顿太碎 2/3
  + 24124112「断句特别快」+ 고구마「不喘气」** → 断句停顿是最硬的信号
- **机器定位到物理根源**：我们「中等停顿 0.35~0.5s」**72 个**，
  EL 标杆只有 **15 个**（差 4.8 倍），累计多占 24.5 秒。
  **「碎」的物理定义就是大量中等长度的口把句子切碎。**

所以本轮：
1. 不再问「哪种闷」，改问**具体可执行的现象**
2. 断句停顿从「主观描述」升级到「**机器已定位的差异**」，问员工能不能听出
3. 新增：고구마 提的「电音严重 / 倍镜噪音严重」——**这两个此前零指标覆盖**
4. 新增：情绪是否「像念稿」（디아 唯一一次选它）

## 🔴 本轮修掉的数据质量问题（第23轮发现）
1. **控制题默认预选 C**（第23轮 고구마 漏答控制题导致整份作废）
   → 页面改用 `preselect` 字段，服务端与前端已支持默认预选。
   ⚠️ 但更稳的做法是**把控制题放最后一题**（已改），并在 note 里写明
   「这题选错了也没关系，但**必须选**」。
2. **题干显式写「必须选一个」**，避免漏答。

## 素材（全部取自 496 秒真长文 longprobe-6cdd）
🔴 用户铁律：短文案验收没意义。这里刻意全部用长篇，
   且**开头/中后段两个位置都切**，因为第23轮 u2 两处答案相反。
"""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
TMPD = "/mnt/tts/diag/_q24"
TGT_LUFS = -26.1
DUR = 30.0

RAW = "/mnt/tts/output/longprobe-6cdd/longprobe-6cdd-1.mp3"   # 当前链路 496s 真长文
# 🔴 EL 用 merged.wav（509s）不是 seg0.wav（248s）——
#    首版用 seg0 导致 380s 起点越界、emit 返回 None 而崩。
#    merged 才是与长文等长的完整标杆，起点才能对齐。
EL_S = "/mnt/tts/data/cases/6cdd25bc75f2/merged.wav"           # 同文案 EL 标杆 509s

# 起点互不重叠（30s 窗口，间隔 ≥30s），全在 496/509s 内
P_E, P_M, P_L = 60.0, 200.0, 380.0


def lufs(p):
    r = subprocess.run(["ffmpeg", "-i", p, "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    v = [l for l in r.stderr.splitlines() if "I:" in l and "LUFS" in l]
    return float(v[-1].split("I:")[1].split("LUFS")[0].strip()) if v else None


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()[:8]


def cut(src, start, dur):
    """切一段。🔴 越界要当场报清楚，不要静默产出空/短文件。

    ⚠️ 临时文件必须建在 /mnt/tts/diag 下（与目标同文件系统）。
    用 tempfile 默认的 /tmp 会**跨设备**，os.replace 报 EXDEV；
    历史上也栽过「临时目录用 /tmp 导致权限/属主冲突」。
    """
    sr = 16000
    t = os.path.join(TMPD, "_t%d.wav" % int(start * 1000))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-ss", "%.1f" % start, "-ar", str(sr), "-ac", "1",
                    "-t", "%.1f" % dur, t], check=True)
    got = os.path.getsize(t) / 2.0 / sr      # 16bit 单声道 = 2 bytes/frame
    if got < dur - 0.5:
        raise SystemExit(
            "🔴 切片越界：%s 第 %.0fs 起取 %.0fs，实际只有 %.1fs\n"
            "   请把起点调到素材长度以内"
            % (os.path.basename(src), start, dur, got))
    return t


def emit(wav, name, title):
    g = TGT_LUFS - lufs(wav)
    mp3 = os.path.join(FB_DIR, "%s.mp3" % name)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav,
                    "-af", "volume=%gdB" % g,          # 🔴 dB 不是 %gdBB
                    "-b:a", "192k", "-metadata", "title=%s" % title, mp3],
                   check=True)
    back = lufs(mp3)
    if back is None:
        # 🔴 读不到 LUFS 通常意味着切片本身坏了（起点越界/空文件）。
        #    首版这里直接 None-float 崩掉，报错难懂；改成明确报错。
        print("  %-10s 🔴 读不出 LUFS（切片可能越界或为空）" % name)
        return None
    print("  %-10s md5=%s  LUFS %6.1f %s"
          % (name, md5(mp3), back, "OK" if abs(back - TGT_LUFS) < 1.0 else "!!爆表"))
    return mp3


def main():
    os.makedirs(TMPD, exist_ok=True)
    made = {}

    for tag, src in (("raw", RAW), ("el", EL_S)):
        for pos, st in (("e", P_E), ("m", P_M), ("l", P_L)):
            nm = "q24_%s_%s" % (tag, pos)
            made[nm] = emit(cut(src, st, DUR), nm, "clip %s" % nm)

    # 控制题（放最后，见文件头说明），起点 150s 与上面不重叠
    ctrl = cut(RAW, 150.0, DUR)
    made["q24_ctrl1"] = emit(ctrl, "q24_ctrl1", "ctrl A")
    made["q24_ctrl2"] = emit(ctrl, "q24_ctrl2", "ctrl B")

    print("\n=== md5 自检 ===")
    inv = {}
    for nm, p in made.items():
        inv.setdefault(md5(p), []).append(nm)
    bad = False
    for h, ns in inv.items():
        if len(ns) > 1:
            allowed = set(ns) <= {"q24_ctrl1", "q24_ctrl2"}
            print("  %-9s %s" % (h, "OK(控制题同素材)" if allowed
                                 else "🔴 撞车 -> %s" % ns))
            if not allowed:
                bad = True
    if not bad:
        print("  ✅ 无撞车")

    items = [
        # ===== 核心：断句停顿（机器已定位差异，问耳朵能不能听出）=====
        {"id": "q1_pause_ours_vs_el_20261015q", "dim": "断句停顿",
         "title": "这两段，哪一段听起来更顺、更像真人讲话？",
         "desc": "两段都是男声，各 30 秒，同一段文案。\n\n"
                 "第 1 段是我们的版本，第 2 段是 ElevenLabs 的。\n\n"
                 "**只判断「顺不顺」，不用管别的。**",
         "ref": None,
         "note": "机器已定位差异：我们「中等停顿 0.35~0.5s」有 72 个，"
                 "EL 只有 15 个（差 4.8 倍），累计多占 24.5 秒。\n"
                 "这题验证：**耳朵能不能听出这种差异**。\n"
                 "A/B 语义：A=我们，B=EL（全轮一致）。",
         "options": [
             {"key": "A", "file": "q24_raw_e.mp3", "label": "第 1 段（我们的）更顺"},
             {"key": "B", "file": "q24_el_e.mp3", "label": "第 2 段（EL）更顺"},
             {"key": "C", "file": None, "label": "差不多"}]},

        # 反向确认：我们的停顿是不是真的"碎/不喘气"
        {"id": "q2_pause_feels_20261015q", "dim": "断句停顿",
         "title": "听我们这段，你觉得主要是哪种问题？",
         "desc": "一段男声，30 秒，是我们现在的版本。\n\n"
                 "**必须选一个**，按最明显的感觉说。",
         "ref": {"label": "▶ 点这个播放", "file": "q24_raw_m.mp3",
                 "note": "我们的版本。"},
         "note": "第23轮 u7 拿到「②停顿太多太碎」2/3，此处复核并细分。\n"
                 "机器侧对应：中等停顿(0.35~0.5s) 多 57 个。\n"
                 "预期 ①（碎）与机器发现一致。",
         "options": [
             {"key": "A", "file": None, "label": "① 停顿太多太碎，一句被切得很碎"},
             {"key": "B", "file": None, "label": "② 停顿太少，听起来喘不上气"},
             {"key": "C", "file": None, "label": "③ 停顿位置不对，该断的地方没断"},
             {"key": "D", "file": None, "label": "④ 停顿这块没什么问题"}]},

        # ===== 고구마 提的「电音严重」——此前零指标覆盖 =====
        {"id": "q3_electronic_20261015q", "dim": "电音感",
         "title": "这段听起来像不像真人录的？还是像电子合成？",
         "desc": "一段男声，30 秒，是我们现在的版本。\n\n"
                 "就判断「像不像机器合成的」。",
         "ref": {"label": "▶ 点这个播放", "file": "q24_raw_m.mp3",
                 "note": "我们的版本。"},
         "note": "고구마 自由文本：「电音严重」。\n"
                 "这个维度**此前完全没有指标覆盖**，是本轮新增探针。\n"
                 "若多数人选「像电子合成」，说明要查底噪合成感 / 过净(flat 曾 −2.61σ)。",
         "options": [
             {"key": "A", "file": None, "label": "① 像真人录的"},
             {"key": "B", "file": None, "label": "② 有点电子感，但还能接受"},
             {"key": "C", "file": None, "label": "③ 明显像电子合成的"}]},

        # ===== 고구마 提的「倍镜噪音严重」→ 高频噪点 =====
        {"id": "q4_noise_20261015q", "dim": "底噪与噪点",
         "title": "这段里有没有明显让你不舒服的杂音？",
         "desc": "一段男声，30 秒，是我们现在的版本。\n\n"
                 "戴耳机或安静环境听更容易听出来。",
         "ref": {"label": "▶ 点这个播放", "file": "q24_raw_m.mp3",
                 "note": "我们的版本。建议戴耳机。"},
         "note": "고구마 提「倍镜噪音严重」（应为「背景噪音严重」）。\n"
                 "机器侧对应：4k~8k 沿时间轴极差 **94.9%**（EL 只有 31.9%）。\n"
                 "预期听到「嘶嘶/沙沙/电流声」的人较多 → 高频稳定性是真问题。",
         "options": [
             {"key": "A", "file": None, "label": "① 有明显的嘶嘶声/沙沙声"},
             {"key": "B", "file": None, "label": "② 有电流声/电音杂音"},
             {"key": "C", "file": None, "label": "③ 背景有底噪沙沙声"},
             {"key": "D", "file": None, "label": "④ 没听出杂音，很干净"}]},

        # ===== 「情绪太平像念稿」（디아 上一轮唯一选它）=====
        {"id": "q5_monotone_20261015q", "dim": "情绪与语气",
         "title": "这段听起来像在「念稿」，还是像在「跟人讲话」？",
         "desc": "一段男声，30 秒，是我们现在的版本。\n\n"
                 "就判断语气像不像在跟人说话。",
         "ref": {"label": "▶ 点这个播放", "file": "q24_raw_l.mp3",
                 "note": "我们的版本。"},
         "note": "디아 在第23轮 u5 唯一选了「⑤情绪太平，像在念稿」，"
                 "这一票单独出题求证。\n"
                 "机器侧**无指标覆盖** → 若多数人认同，需要新建情绪维度。",
         "options": [
             {"key": "A", "file": None, "label": "① 像在念稿，语气很平"},
             {"key": "B", "file": None, "label": "② 介于两者之间"},
             {"key": "C", "file": None, "label": "③ 像在跟人讲话，有起伏"}]},

        # ===== 长文案一致性（第23轮「开头更好 2/3」）=====
        {"id": "q6_consistency_20261015q", "dim": "长文案一致性",
         "title": "同一条音频的开头和中后段，哪一段更自然？",
         "desc": "两段都来自同一条音频，各 30 秒。\n"
                 "**必须选一个。**",
         "ref": None,
         "note": "第23轮 u3 拿到「开头更好 2/3」，机器侧也显示后半段在掉"
                 "（沿时间轴极差 47% vs EL 12~16%）。此题复核。",
         "options": [
             {"key": "A", "file": "q24_raw_e.mp3", "label": "开头那段（60s）"},
             {"key": "B", "file": "q24_raw_l.mp3", "label": "中后段那段（380s）"},
             {"key": "C", "file": None, "label": "差不多"}]},

        # ===== 最终验收：能否直接发客户（中后段）=====
        {"id": "q7_send_l_20261015q", "dim": "可发送性",
         "title": "中后段这 30 秒，哪一段可以直接发给客户？",
         "desc": "两段都是男声，各 30 秒，同一段文案。\n\n"
                 "按「能不能直接发给客户」判断。",
         "ref": None,
         "note": "**这是唯一真正的验收标准。**\n"
                 "第23轮两处位置都是 3/3 选 EL，本题看是否仍如此。\n"
                 "A=我们，B=EL。",
         "options": [
             {"key": "A", "file": "q24_raw_l.mp3", "label": "第 1 段（我们的）"},
             {"key": "B", "file": "q24_el_l.mp3", "label": "第 2 段（EL）"},
             {"key": "C", "file": None, "label": "都不行"}]},

        # ===== 优先级（第23轮 2/3 选断句停顿，这里复核排序是否变）=====
        {"id": "q8_priority_20261015q", "dim": "优先级排序",
         "title": "如果只能改一处，你最想先改哪里？",
         "desc": "先听完上面那段（我们的版本），再选。\n\n"
                 "**只选一个**——如果只能改一处，你会改哪处？",
         "ref": {"label": "▶ 再听一次", "file": "q24_raw_m.mp3",
                 "note": "我们的版本。"},
         "note": "第23轮此题 2/3 选「断句停顿」，디아 选「情绪太平像念稿」。\n"
                 "**三人共识的选项 = 最高优先级**，直接决定下一轮动手顺序。",
         "options": [
             {"key": "A", "file": None, "label": "① 断句停顿不自然"},
             {"key": "B", "file": None, "label": "② 情绪太平，像在念稿"},
             {"key": "C", "file": None, "label": "③ 有杂音/底噪"},
             {"key": "D", "file": None, "label": "④ 像电子合成的"},
             {"key": "E", "file": None, "label": "⑤ 前后忽明忽暗，不一致"},
             {"key": "F", "file": None, "label": "⑥ 都不突出，说不上来"}]},

        # ===== 控制题放最后（第23轮 고구마 漏答 → 整份作废）=====
        {"id": "ctrl_pace_20261015q", "dim": "答题有效性校验",
         "title": "【最后一题】这两段，哪一段念得更快？",
         "desc": "**这道题必须选一个**（选错了也没关系，不影响其他题）。\n\n"
                 "同一段音频放了两遍，各 30 秒。听 5 秒左右就够。",
         "ref": None,
         "note": "隐藏控制题：q24_ctrl1/q24_ctrl2 同素材，正解=C 一样快。\n"
                 "选 A/B 说明没在听，这一整份作废。\n"
                 "🔴 放最后是因为第23轮 고구마 漏答控制题导致整份白填——"
                 "漏答成本太高，宁可牺牲「第一题就暴露」的早期预警。",
         "options": [
             {"key": "A", "file": "q24_ctrl1.mp3", "label": "第 1 段念得更快"},
             {"key": "B", "file": "q24_ctrl2.mp3", "label": "第 2 段念得更快"},
             {"key": "C", "file": None, "label": "两段一样快"}]},
    ]

    print("\n=== 题目完整性 ===")
    empty = []
    for it in items:
        hf = any(o.get("file") for o in it.get("options") or [])
        hr = bool(it.get("ref") and it["ref"].get("file"))
        if not (hf or hr):
            empty.append(it["id"])
        print("  %-32s 选项=%s 参考=%s"
              % (it["id"], "有" if hf else "无", "有" if hr else "无"))
    print("  ✅ 每题都有音频" if not empty else "  🔴 废题: %s" % empty)

    p = os.path.join(FB_DIR, "items.json")
    if os.path.exists(p):
        shutil.copyfile(p, p + ".bak-before-q")
        print("\n已备份旧题库 ->", p + ".bak-before-q")
    json.dump({"items": items}, open(p, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("题库已写入 %s  共 %d 题" % (p, len(items)))
    if bad or empty:
        print("🔴 存在拦截项，先别让员工答题")
    print("\n发布后请确认：curl -s http://服务器:8000/api/fb/where")


if __name__ == "__main__":
    main()