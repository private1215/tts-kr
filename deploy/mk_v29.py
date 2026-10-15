#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第二十九轮（v）题库：**验证 declick 是否消除了「卡顿」**（闭环 ③④）

## 这一轮验证的修复
第 28 轮两名员工都说「还是存在声音卡顿」（2/2）。
🔴 已定位真根因并修复上线：

| | 最陡落差 | 硬切(>60dB) | 全零帧 |
|---|---|---|---|
| **EL 标杆** | **−25.3 dB** | **0** | **0 (0.0%)** |
| 修复前（OmniVoice） | −101.3 dB | 10 | 2 |
| **修复后** | **−30.7 dB** | **0** | **0** |

机制：我们输出的停顿电平低到 **−85~−146 dBFS（近乎绝对零）**，
而真人停顿是低电平但**有环境声** → 听感就是「咔」。
`declick` 给所有峰值 < −60 dBFS 的 1 ms 块补 −70 dBFS 极轻噪声。

🔴 **换引擎（vox→omni）也一并验过了**：第 27 轮 2 人答案相反
   （一人「换之后更自然」、一人「差不多」），**没有明确结论**。
   这轮把「卡顿」和「电子感」分开问，各自有明确判读。

## 本轮 3 题，每题都对应一个动作
| 题 | 答案 → 动作 |
|---|---|
| v1 卡顿是否消失 | 消失 → declick 有效，保留；还在 → 查编码器/改用 wav 输出 |
| v2 电子感是否改善 | 改善 → 换引擎+declick 有效；没改善 → 是架构限制 |
| v3 能否直接发客户 | 长期基线锚点（此前 15 票全选 EL）|

🔴 **控制题沿用第 28 轮的「新问法」**（「这两段听起来有区别吗」）——
   实测 2/2 全对，而旧问法 4 人里 2 人错。**有效就保留，不折腾。**

## 素材（用户铁律：全部 ≥180s 真长文，同一份 6cdd 文案）
- 修复后：/mnt/tts/output/longprobe-declick-6cdd/（declick 后的新版本）
- 修复前：/mnt/tts/output/longprobe-omni-1block/（卡顿版本）
- EL 标杆：/mnt/tts/data/cases/6cdd25bc75f2/merged.wav
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
TMPD = "/mnt/tts/diag/_v29"
TGT_LUFS = -26.1
DUR = 30.0

NEW_ = "/mnt/tts/output/longprobe-declick-6cdd/longprobe-declick-6cdd-1.mp3"
OLD_ = "/mnt/tts/output/longprobe-omni-1block/longprobe-omni-1block-1.mp3"
EL_S = "/mnt/tts/data/cases/6cdd25bc75f2/merged.wav"

# 🔴 位置选择：**避开 330s 附近**。
#    第 28 轮 디아 说「EL 标杆那段也有卡顿」——说明 330s 起那一段
#    连 EL 标杆本身都有可听接缝，拿它做对照会两边都卡、无法分辨。
#    改用 100s / 220s / 440s 三处，都是干净区间。
P_E, P_M, P_L = 100.0, 220.0, 440.0


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
        sys.exit("🔴 切片越界：%s 第%.0fs起取%.0fs，实际仅%.1fs\n"
                 "   请把起点调到素材长度以内"
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
    for p in (NEW_, OLD_, EL_S):
        if not os.path.exists(p):
            print("🔴 缺素材：%s" % p)
            print("   （新素材要等 longprobe-declick-6cdd 生成完）")
            return 1
    os.makedirs(TMPD, exist_ok=True)
    made = {}
    for tag, src in (("new", NEW_), ("old", OLD_), ("el", EL_S)):
        for pos, st in (("e", P_E), ("m", P_M), ("l", P_L)):
            nm = "v29_%s_%s" % (tag, pos)
            made[nm] = emit(cut(src, st, DUR), nm, "clip %s" % nm)

    ctrl = cut(NEW_, 300.0, DUR)
    made["v29_ctrl1"] = emit(ctrl, "v29_ctrl1", "ctrl A")
    made["v29_ctrl2"] = emit(ctrl, "v29_ctrl2", "ctrl B")

    print("\n=== md5 自检 ===")
    inv = {}
    for nm, p in made.items():
        inv.setdefault(md5(p), []).append(nm)
    bad = False
    for h, ns in inv.items():
        if len(ns) > 1:
            ok = set(ns) <= {"v29_ctrl1", "v29_ctrl2"}
            print("  %-9s %s" % (h, "OK(控制题同素材)" if ok
                                 else "🔴 撞车 -> %s" % ns))
            if not ok:
                bad = True
    if not bad:
        print("  ✅ 无撞车")

    items = [
        # ---- 1. 核心：卡顿是否被消除 ----
        {"id": "v1_stuck_fixed_20261015v", "dim": "declick 验证",
         "title": "这两段，哪一段有「突然卡一下 / 吃字」的地方？",
         "desc": "两段都是男声，同一段文案，各 30 秒。\n\n"
                 "**请仔细听有没有哪个字、哪个瞬间是突然卡住的。**\n"
                 "如果两段都听不出，就选最后一项。",
         "ref": None,
         "note": "🔴🔴 本轮核心题，验证刚上线的 declick。\n"
                 "第1段=已修复（补过极安静段的环境声）\n"
                 "第2段=修复前（停顿电平低到 -101dB，听感是「咔」）\n\n"
                 "机器侧：修复后硬切 10→0、最陡 −101.3dB→−30.7dB，\n"
                 "时长零变化（不改语速）。EL 标杆是 −25.3dB / 0 处硬切。\n\n"
                 "选A(新版本卡) → declick 没解决，得查 MP3 编码器或改输出 wav\n"
                 "选B(旧版本卡) → 修复被证实\n"
                 "选C(都听不出) → 修复有效，且说明之前听到的是别的问题",
         "options": [
             {"key": "A", "file": "v29_new_m.mp3", "label": "第 1 段（新版本）有卡顿"},
             {"key": "B", "file": "v29_old_m.mp3", "label": "第 2 段（旧版本）有卡顿"},
             {"key": "C", "file": None, "label": "两段都听不出卡顿"}]},

        # ---- 2. 电子感（换引擎 + declick 之后是否改善）----
        {"id": "v2_electronic_now_20261015v", "dim": "电子感（复验）",
         "title": "这段现在听起来，像真人录的还是像电子合成的？",
         "desc": "一段男声，30 秒，是**现在线上的版本**（已换引擎 + 已修复卡顿）。\n\n"
                 "就按「像真人还是像电子合成」判断。",
         "ref": {"label": "▶ 点这个播放", "file": "v29_new_e.mp3",
                 "note": "现在线上的版本。"},
         "note": "🔴 第26轮 s1 拿到 3/3「完全不能接受」，第27轮两人相反\n"
                 "（「有点合成能接受」vs「还是明显像电子合成」）→ **问不出信息量**。\n"
                 "现在换引擎(vox→omni)+declick 两项都上了，**再问一次才有意义**。\n"
                 "选③ → 是架构限制，换引擎也不够，得考虑非自回归方案或微调。",
         "options": [
             {"key": "A", "file": None, "label": "① 像真人录的"},
             {"key": "B", "file": None, "label": "② 有点合成感，但能接受"},
             {"key": "C", "file": None, "label": "③ 还是明显像电子合成"}]},

        # ---- 3. 长期基线锚点 ----
        {"id": "v3_send_baseline_20261015v", "dim": "可发送性",
         "title": "这 30 秒，哪一段可以直接发给客户？",
         "desc": "两段都是男声，同一段文案，各 30 秒。\n\n"
                 "按「能不能直接发给客户」判断。",
         "ref": None,
         "note": "**长期基线锚点。**\n"
                 "此前 4 轮共 **15 张票无一例外全选 EL**。\n"
                 "→ 哪天出现「我们也能发」，才算真有进展。",
         "options": [
             {"key": "A", "file": "v29_new_l.mp3", "label": "第 1 段（我们的）"},
             {"key": "B", "file": "v29_el_l.mp3", "label": "第 2 段（EL）"},
             {"key": "C", "file": None, "label": "都不行"}]},
    ]

    # 控制题沿用第 28 轮的新问法（实测有效）
    items.append(
        {"id": "ctrl_same_20261015v", "dim": "答题有效性校验",
         "title": "【最后一题】这两段，听起来有区别吗？",
         "desc": "**这道题必须选一个**（选错也没关系）。\n\n"
                 "这两段是**同一段音频放了两遍**，内容完全一样。\n"
                 "听 5 秒就够。",
         "ref": None,
         "note": "隐藏控制题：正解=C 两段一样。\n"
                 "🔴 沿用第28轮问法「这两段听起来有区别吗」——\n"
                 "   实测 2/2 全对，而旧问法「哪段念得更快」是 4 人里 2 人错。\n"
                 "   有效就保留，不折腾。",
         "options": [
             {"key": "A", "file": "v29_ctrl1.mp3", "label": "第 1 段不一样"},
             {"key": "B", "file": "v29_ctrl2.mp3", "label": "第 2 段不一样"},
             {"key": "C", "file": None, "label": "两段一样，听不出区别"}]})

    print("\n=== 题目完整性 ===")
    empty = []
    for it in items:
        hf = any(o.get("file") for o in it.get("options") or [])
        hr = bool(it.get("ref") and it["ref"].get("file"))
        if not (hf or hr):
            empty.append(it["id"])
        print("  %-30s 选项=%s 参考=%s"
              % (it["id"], "有" if hf else "无", "有" if hr else "无"))
    print("  ✅ 每题都有音频" if not empty else "  🔴 废题: %s" % empty)

    p = os.path.join(FB_DIR, "items.json")
    if os.path.exists(p):
        shutil.copyfile(p, p + ".bak-before-v")
        print("\n已备份旧题库 ->", p + ".bak-before-v")
    json.dump({"items": items}, open(p, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("题库已写入 %s  共 %d 题" % (p, len(items)))
    if bad or empty:
        print("🔴 存在拦截项，先别让员工答题")
    print("\n发布后确认：curl -s http://服务器:8000/api/fb/where")
    print("归档：python3 tools/fb_loop.py --archive v")


if __name__ == "__main__":
    sys.exit(main() or 0)