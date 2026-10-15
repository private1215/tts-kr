#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第三十轮（w）题库：**男声修好了 + 卡顿是否消除**（闭环 ⑤）

## 这一轮为什么重要
上一轮（v）结束后排查发现两个**足以毁掉全部反馈结论**的问题：

### 1. 🔴 男声一直在被当女声处理
第 29 轮 `swap_engine.py` 把 `EL_male.engine` 从 `voxcpm` 换成 `omnivoice`，
但代码里两处**按引擎名判性别**：
- `_run_fix_chain`：`--gender "m" if _engine_name == "voxcpm" else "f"`
  → 男声被按女声处理：F0 目标 276Hz（男声应 197.7）、语速锚 7.57（应 8.37）
- `_synth_omni`：`voice="female"` 写死
  → 而 **OmniVoice `/health` 只有 `["female"]`，根本没有男声模型**

实测（4143 字同一文案）：
| | F0 中位 | 字/秒 |
|---|---|---|
| EL 男标杆 6cdd | **179.9 Hz** | 8.14 |
| 我们的「男声」（换引擎后） | **277.3 Hz** | 7.43 |
| 我们的女声 | 277.5 Hz | 7.45 |

→ **第 28/29 轮员工实际在拿「我们的女声」跟「EL 男声」比，反馈被污染。**
✅ 已回退 `EL_male.engine=voxcpm`、`cps=8.37`，并把性别判定改为按音色配置
   （`deploy/apply_gender_fix.py`）。本轮是**修好之后的第一轮**。

### 2. ✅ declick 第30轮升级：逐样本「幅度下限」
老男声（voxcpm）有 **165 个数字零帧 / 15 处硬切**（女声只有 17 / 9）。
| | 硬切 | 最陡5帧落差 | 真零帧 |
|---|---|---|---|
| 老男声（无 declick） | 15 | — | 165 |
| EL 标杆 | 1 | −55.86 dB | 1 |
新 declick：门限 −100dB（只认真数字零）、填充 −89dB（=EL 静音帧 p5）、
逐样本 `where(|噪|>|原|, 噪, 原)` —— 原信号比噪声响的地方一个字节都不动。

## 本轮 3 题 + 控制题，每题都对应一个动作
| 题 | 答案 → 动作 |
|---|---|
| w1 卡顿是否消除 | ②→declick 有效；①/③→还有别的卡顿源，查编码器/改 wav 输出 |
| w2 最不像真人的地方 | ①→换音色方案；②→调韵律；③→调停顿；④→调语速；⑤→查链路 |
| w3 能否直接发客户 | 长期基线锚点（此前 15 票全选 EL）|

## 素材（用户铁律：全部 ≥180s 真长文，同一份 6cdd 文案，全是男声）
- 新：`longprobe-r30vox-6cdd`（voxcpm + 第30轮 declick）
- 旧：`longprobe-6cdd`（voxcpm，无 declick，165 个数字零帧）
- EL：`/mnt/tts/data/cases/6cdd25bc75f2/merged.wav`（509s）
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
TMPD = "/mnt/tts/diag/_w30"
TGT_LUFS = -26.1
DUR = 30.0

NEW_ = "/mnt/tts/output/longprobe-r30vox-6cdd-3/longprobe-r30vox-6cdd-1.mp3"
OLD_ = "/mnt/tts/output/longprobe-6cdd/longprobe-6cdd-1.mp3"
EL_S = "/mnt/tts/data/cases/6cdd25bc75f2/merged.wav"

# 🔴 位置选择：**必须落在旧源确实有硬切的地方**（第29轮的教训）。
#    实测旧男声 15 处硬切位置(s)：
#      10.0 44.4 70.9 140.2 200.7 200.9 207.1 342.9 343.2 366.0 412.7 413.0
#    → 取 188.0s 起（188~218s），一段里就含 200.7 / 200.9 / 207.1 三处。
#    EL 标杆自身在 356.7s 有 1 处接缝 → 避开 330~370s。
P_HARD = 188.0     # w1：旧源有 3 处硬切
P_BASE = 250.0     # w4：基线题，避开 EL 的 356.7s 接缝
P_REF = 120.0      # w2/w3：音高 A/B 与单段定位题

# 🔴 第30轮新发现：男声音高偏高。
#    同一套 parselmouth 口径实测：EL 男标杆 179.9Hz，我们的成品 **210.1Hz（高 17%）**。
#    （老版本只有 186.3Hz —— 因为那时男声被当女声处理，变调逻辑没按男声走。）
#    候选：用 asetrate 三步把音高降到 EL 的水平（记忆铁律：变调只用 asetrate，
#    禁用 rubberband；三步缺一不可）。
PITCH_RATIO = 179.9 / 210.1   # ≈0.856


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
        sys.exit("🔴 切片越界：%s 第%.0fs起取%.0fs，实际仅%.1fs"
                 % (os.path.basename(src), start, dur, got))
    return t


def emit_pitch(wav, name, ratio):
    """变调：asetrate + aresample + atempo 三步（记忆铁律，缺一不可）。

    漏了 atempo → 时长被压短 1/R。这里 R<1（降调），时长会被拉长 1/R。
    """
    out = os.path.join(TMPD, "_%s.wav" % name)
    sr = 16000
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav,
                    "-af",
                    "asetrate=%d,aresample=%d,atempo=%.6f"
                    % (int(round(sr * ratio)), sr, 1.0 / ratio),
                    "-ar", str(sr), "-ac", "1", out], check=True)
    return out


def emit(wav, name, title, normalise=True, as_wav=False):
    """🔴 第29轮的坑：验证「停顿电平」的对照题必须 normalise=False，
    响度归一会把极低电平整体抬平，两个选项就没差异了。
    🔴 另一个坑：`volume=%gdB` 不是 `%gdBB`（多一个 B = +18dB）。
    """
    g = (TGT_LUFS - lufs(wav)) if normalise else 0.0
    ext = "wav" if as_wav else "mp3"
    out = os.path.join(FB_DIR, "%s.%s" % (name, ext))
    af = ["-af", "volume=%gdB" % g] if normalise else []
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav] + af +
                   ["-b:a", "192k", "-metadata", "title=%s" % title, out],
                   check=True)
    back = lufs(out)
    if back is None:
        print("  %-14s 🔴 读不出 LUFS" % name)
        return None
    print("  %-14s md5=%s  LUFS %6.1f %s"
          % (name, md5(out), back,
             "OK" if abs(back - TGT_LUFS) < 1.0 else "!!爆表"))
    return out


def main():
    for p in (NEW_, OLD_, EL_S):
        if not os.path.exists(p):
            print("🔴 缺素材：%s" % p)
            return 1
    os.makedirs(TMPD, exist_ok=True)
    made = {}

    # w1：卡顿对照（wav + 归一）
    #    🔴 用 wav 是因为 **mp3 才会抹掉硬切**（libmp3lame 把 −154.9dB 抬到 −41.1dB）。
    #    🔴 归一化用的是**静态** `volume=XdB`，只改整体增益、相对落差不变，
    #       所以不会抹掉硬切（旧记忆里「归一抹掉硬切」其实是 mp3 造成的，已实测纠正）。
    #       归一后两个选项响度一致，员工才不会按「哪个声音大」来选。
    made["w30_new_h"] = emit(cut(NEW_, P_HARD, DUR), "w30_new_h", "clip new",
                             normalise=True, as_wav=True)
    made["w30_old_h"] = emit(cut(OLD_, P_HARD, DUR), "w30_old_h", "clip old",
                             normalise=True, as_wav=True)
    # w2：音高 A/B（候选 = 降到 EL 男声的 179.9Hz）
    _ref = cut(NEW_, P_REF, DUR)
    made["w30_new_r"] = emit(_ref, "w30_new_r", "clip ref")
    made["w30_low_r"] = emit(emit_pitch(_ref, "w30_low_r", PITCH_RATIO),
                             "w30_low_r", "clip low", normalise=True,
                             as_wav=False)
    # w3：基线题（归一）
    made["w30_new_b"] = emit(cut(NEW_, P_BASE, DUR), "w30_new_b", "clip base")
    made["w30_el_b"] = emit(cut(EL_S, P_BASE, DUR), "w30_el_b", "clip el")

    ctrl = cut(NEW_, P_REF, DUR)
    made["w30_ctrl1"] = emit(ctrl, "w30_ctrl1", "ctrl A")
    made["w30_ctrl2"] = emit(ctrl, "w30_ctrl2", "ctrl B")

    print("\n=== md5 自检 ===")
    inv = {}
    for nm, p in made.items():
        inv.setdefault(md5(p), []).append(nm)
    bad = False
    for h, ns in inv.items():
        if len(ns) > 1:
            ok = set(ns) <= {"w30_ctrl1", "w30_ctrl2"}
            print("  %-9s %s" % (h, "OK(控制题同素材)" if ok
                                 else "🔴 撞车 -> %s" % ns))
            if not ok:
                bad = True
    if not bad:
        print("  ✅ 无撞车")

    items = [
        # ---- 1. 核心：卡顿是否被消除 ----
        {"id": "w1_stuck_fixed_20261015w", "dim": "declick 验证（男声）",
         "title": "这两段，哪一段有「突然卡一下 / 吃字」的地方？",
         "desc": "两段都是男声，同一段文案，各 30 秒。\n\n"
                 "**第 1 段和第 2 段都是我们的版本**，只是处理方式不同。\n"
                 "**请仔细听有没有哪个字、哪个瞬间是突然卡住的。**\n"
                 "两段都卡就选「两段都有」，都不卡就选「两段都没有」。",
         "ref": None,
         "note": "🔴🔴 本轮核心题。\n"
                 "第1段=新版（补过数字零段的环境声）\n"
                 "第2段=旧版（停顿电平掉到 −200dB 以下，听感是「咔」）\n\n"
                 "机器侧（10ms 帧口径）：\n"
                 "  旧男声：硬切 15 处、真零帧 165 个\n"
                 "  EL标杆：硬切 1 处、真零帧 1 个\n"
                 "  新版：硬切 0、真零帧 0\n"
                 "🔴 素材位置 188s（188~218s）：旧源这一段里有 200.7 / 200.9 / 207.1\n"
                 "   三处硬切；首版若选在无硬切处，两段就没差异（第29轮踩过）。\n"
                 "🔴 这两个选项是 **wav 不是 mp3**：libmp3lame 会把硬切\n"
                 "   从 −154.9dB 抹到 −41.1dB，用 mp3 做对照根本测不出。\n"
                 "🔴 也不做响度归一：归一会把停顿极低电平整体抬平，同样抹掉差异。\n\n"
                 "选①第1段卡 → declick 反而有害，回退\n"
                 "选②第2段卡 → ✅ 修复被证实，保留\n"
                 "选③两段都卡 → 还有别的卡顿源，查编码器/考虑改 wav 输出\n"
                 "选④两段都不卡 → 卡顿已解决（或这段听不出），关闭该方向",
         "options": [
             {"key": "A", "file": "w30_new_h.wav", "label": "① 第 1 段有卡顿"},
             {"key": "B", "file": "w30_old_h.wav", "label": "② 第 2 段有卡顿"},
             {"key": "C", "file": None, "label": "③ 两段都有卡顿"},
             {"key": "D", "file": None, "label": "④ 两段都没有卡顿"}]},

        # ---- 2. 音高 A/B（第30轮新发现）----
        {"id": "w2_pitch_ab_20261015w", "dim": "音高（男声）",
         "title": "这两段男声，哪一段的声音更自然？",
         "desc": "两段都是我们的男声，**念的内容完全一样，只有声音的高低不同**。\n\n"
                 "就按「哪个更像真人说话的声音」判断，不用管别的。\n"
                 "两段都行就选「两段都自然」，都不行就选「两段都不自然」。",
         "ref": None,
         "note": "🔴🔴 第30轮新发现：**我们的男声音高比 EL 高 17%**。\n"
                 "同一套 parselmouth 口径实测 F0 中位：\n"
                 "  EL 男标杆 6cdd    **179.9 Hz**\n"
                 "  我们（当前线上）   **210.1 Hz**  ← 高 17%\n"
                 "  老版本（第29轮前）  186.3 Hz\n"
                 "→ 老版本其实更接近，只是那时男声被当女声处理、变调没按男声走。\n\n"
                 "第1段 = 当前音高（210Hz）\n"
                 "第2段 = 候选：降到 EL 的水平（179.9Hz，ratio=0.856，\n"
                 "        走 asetrate+aresample+atempo 三步，时长不变）\n\n"
                 "选① → 保持现状，音高不是问题\n"
                 "选② → ✅ 把男声音高降下来上线（改 fix_chain 的 --target-f0）\n"
                 "选③ → 音高不是瓶颈，别在这上面花时间\n"
                 "选④ → 方向不对（可能要再往上或换个思路）",
         "options": [
             {"key": "A", "file": "w30_new_r.mp3", "label": "① 第 1 段更自然"},
             {"key": "B", "file": "w30_low_r.mp3", "label": "② 第 2 段更自然"},
             {"key": "C", "file": None, "label": "③ 两段都自然"},
             {"key": "D", "file": None, "label": "④ 两段都不自然"}]},

        # ---- 3. 最不像真人的地方（定位）----
        {"id": "w3_whats_off_20261015w", "dim": "不像真人的原因定位",
         "title": "这段男声，最不像真人的地方是哪一点？",
         "desc": "先点下面的播放键听一遍（约 30 秒），然后选**最突出**的那一点。\n\n"
                 "如果同时有几处，就选**你最先注意到**的那个。",
         "ref": {"label": "▶ 点这个播放", "file": "w30_new_r.mp3",
                 "note": "现在线上的男声（已换回 voxcpm + 修复卡顿）。"},
         "note": "🔴 出题闸：每个选项都对应一个明确的下一步动作，没有「说不上来」。\n"
                 "  ① → 换音色方案 / 换参考音频\n"
                 "  ② → 调韵律（F0 曲线、段落能量）\n"
                 "  ③ → 调停顿（断句位置与长短）\n"
                 "  ④ → 调语速（字/秒锚，现按 EL 男声 8.37 对齐）\n"
                 "  ⑤ → 查链路（杂音/吞字/卡顿）\n"
                 "🔴 上一轮（v2）只问「像真人还是电子合成」，两人答案相反\n"
                 "   （一人「像真人」、一人「明显电子合成」）→ 没有可操作的信息。\n"
                 "   这轮改成定位到具体维度。",
         "options": [
             {"key": "A", "file": None, "label": "① 声音本身的质感（发闷 / 发假 / 电子味）"},
             {"key": "B", "file": None, "label": "② 念稿的腔调（起伏太平，像在读稿）"},
             {"key": "C", "file": None, "label": "③ 断句和停顿（位置或长短不对）"},
             {"key": "D", "file": None, "label": "④ 语速（太快或太慢）"},
             {"key": "E", "file": None, "label": "⑤ 有杂音 / 吞字 / 卡顿"}]},

        # ---- 4. 长期基线锚点 ----
        {"id": "w4_send_baseline_20261015w", "dim": "可发送性",
         "title": "这 30 秒，哪一段可以直接发给客户？",
         "desc": "两段都是男声，同一段文案，各 30 秒。\n\n"
                 "按「能不能直接发给客户」判断。",
         "ref": None,
         "note": "**长期基线锚点。**\n"
                 "此前 4 轮共 **15 张票无一例外全选 EL**。\n"
                 "🔴 但那几轮我们的「男声」其实是女声（F0 277Hz vs EL 180Hz），\n"
                 "   参考价值打折。本轮是**修好男声之后的第一票**，格外重要。\n"
                 "→ 哪天出现「我们也能发」，才算真有进展。",
         "options": [
             {"key": "A", "file": "w30_new_b.mp3", "label": "第 1 段（我们的）"},
             {"key": "B", "file": "w30_el_b.mp3", "label": "第 2 段（EL）"},
             {"key": "C", "file": None, "label": "都不行"}]},
    ]

    items.append(
        {"id": "ctrl_same_20261015w", "dim": "答题有效性校验",
         "title": "【最后一题】这两段，听起来有区别吗？",
         "desc": "**这道题必须选一个**（选错也没关系）。\n\n"
                 "这两段是**同一段音频放了两遍**，内容完全一样。\n"
                 "听 5 秒就够。",
         "ref": None,
         "note": "隐藏控制题：正解=C 两段一样。\n"
                 "🔴 沿用第28轮问法「这两段听起来有区别吗」——\n"
                 "   已连续 3 轮全对；旧问法「哪段念得更快」4 人里 2 人错。\n"
                 "   有效就保留，不折腾。",
         "options": [
             {"key": "A", "file": "w30_ctrl1.mp3", "label": "第 1 段不一样"},
             {"key": "B", "file": "w30_ctrl2.mp3", "label": "第 2 段不一样"},
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
        shutil.copyfile(p, p + ".bak-before-w")
        print("\n已备份旧题库 ->", p + ".bak-before-w")
    json.dump({"items": items}, open(p, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("题库已写入 %s  共 %d 题" % (p, len(items)))
    if bad or empty:
        print("🔴 存在拦截项，先别让员工答题")
    print("\n发布后确认：curl -s http://服务器:8000/api/fb/where")
    print("归档：python3 tools/fb_loop.py --archive w")


if __name__ == "__main__":
    sys.exit(main() or 0)
