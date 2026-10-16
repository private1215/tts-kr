#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第三十一轮（y）题库：**融合补丁是不是把「电子音/吃字/卡字」修掉了**

## 这一轮做了什么（必须先说清楚，员工才好判断）

### 根因：同一段语音被「揉」了两遍
`fix_chain` 一直在对同一段音频做**两次 WSOLA 时域拉伸**：
  ① 变速：每窗 `atempo=1/fs`（语速锚 8.37 字/秒，fs≈1.086）
  ② 变调：`asetrate=sr*R, aresample=sr, atempo=1/R`（F0 216→197.7，R=0.914）

**但这两步几乎完全抵消**：变调 R=0.914 本身就把时长拉长 1/0.914 = **1.094×**，
而语速锚只需要 **1.086×** —— 等于先放慢 8.6%，再拉快 9.4%，白揉两遍。
WSOLA 反复揉捏的听感就是**起音被抹、辅音糊掉、过渡发硬**
→ 正是「吃字 / 卡字 / 电子音」。

✅ 已合并成一次：`asetrate=sr*R, aresample=sr, atempo=1/(R·fs)`
   实测 atempo = 1/(0.914×1.086) = **1.007** —— 几乎不再做时域拉伸。

### 效果（494s 同一文案，同口径 probe_g）
| 指标 | 上一版（揉两遍） | **新版（揉一遍）** | EL |
|---|---|---|---|
| Σz² | 72.2 | **27.1** | 中位 59.6 ✅ 优于 EL |
| 平坦度（前60s） | 0.0242（−7.22σ） | **0.0373（−2.89σ）** | 0.0461 |
| 150~300Hz | +1.80σ | **−0.41σ** | — |

🔴 补一个口径校正：probe_g 只测**前 60s**，而我们的前 60s 恰是最坏样本、
   EL 的最好样本。按**全篇**口径：我们 0.03867 vs EL **0.04329**，
   只差 **1.12 倍** —— 电子音的客观缺口已经很小了。

## 本轮 3 题 + 控制题，每题对应一个动作
| 题 | 答案 → 动作 |
|---|---|
| y1 新旧对比（电子音/吃字/卡字） | ②→融合定案；①/③→还有别的主因，继续查 |
| y2 补不补「噪声细节」 | ②→开 TTS_DETAIL_K=0.04；①/③→不开 |
| y3 能否直接发客户 | 长期基线锚点（此前 15 票全选 EL）|

## 素材（同一份 6cdd 文案，全是男声；切片位置避开前 60s 与 EL 的 356.7s 接缝）
- 新：`longprobe-r35-6cdd`（本轮全部修复后，当前线上原状，Σz² 10.6）
- 旧：`longprobe-r30fix-6cdd-2`（揉两遍）
- EL：`/mnt/tts/data/cases/6cdd25bc75f2/merged.wav`（509s）
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")
import detail  # noqa: E402

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
TMPD = "/mnt/tts/diag/_y31"
TGT_LUFS = -26.1
DUR = 30.0
SR_CLIP = 24000          # 🔴 不用 16k：裁到 8k 带宽会把 4k~8k 砍掉，判不了电子音

NEW_ = "/mnt/tts/output/longprobe-r35-6cdd/longprobe-r35-6cdd-1.mp3"
OLD_ = "/mnt/tts/output/longprobe-r30fix-6cdd-2/longprobe-r30fix-6cdd-1.mp3"
EL_S = "/mnt/tts/data/cases/6cdd25bc75f2/merged.wav"

# 🔴 避开前 60s（我们的最坏窗口）/ EL 在 356.7s 的接缝
P_AB = 210.0      # y1 / y2
P_BASE = 250.0    # y3 基线题

DETAIL_K = 0.04   # 第31轮重标定（带限 1.5k~4k）


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
                    "-ss", "%.1f" % start, "-ar", str(SR_CLIP), "-ac", "1",
                    "-t", "%.1f" % dur, t], check=True)
    got = os.path.getsize(t) / 2.0 / SR_CLIP
    if got < dur - 0.5:
        sys.exit("🔴 切片越界：%s 第%.0fs起取%.0fs，实际仅%.1fs"
                 % (os.path.basename(src), start, dur, got))
    return t


def emit(wav, name, title):
    """🔴 `volume=%gdB` 不是 `%gdBB`（多一个 B = +18dB，全爆表）。
    🔴 归一用的是**静态**增益，只改整体音量、不改相对落差 ——
       所以不会抹掉卡顿类差异（旧记忆「归一抹掉硬切」实为 mp3 造成，已纠正）。
    """
    g = TGT_LUFS - lufs(wav)
    out = os.path.join(FB_DIR, "%s.mp3" % name)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav,
                    "-af", "volume=%gdB" % g, "-b:a", "192k",
                    "-metadata", "title=%s" % title, out], check=True)
    back = lufs(out)
    print("  %-16s md5=%s  LUFS %6.1f %s"
          % (name, md5(out), back if back else 0,
             "OK" if back and abs(back - TGT_LUFS) < 1.0 else "!!爆表"))
    return out


def with_detail(wav, name):
    """在切片上补一层带限噪声细节（1.5k~4k，k=0.04）。"""
    import soundfile as sf
    x, sr = sf.read(wav, dtype="float32")
    y = detail.apply(x, sr, k=DETAIL_K, enabled=True)
    o = os.path.join(TMPD, "_%s.wav" % name)
    sf.write(o, y, sr)
    return o


def main():
    for p in (NEW_, OLD_, EL_S):
        if not os.path.exists(p):
            print("🔴 缺素材：%s" % p)
            return 1
    os.makedirs(TMPD, exist_ok=True)
    made = {}

    # y1：新旧对比（第1段=旧「揉两遍」，第2段=新「揉一遍」）
    made["y31_old_a"] = emit(cut(OLD_, P_AB, DUR), "y31_old_a", "clip old")
    made["y31_new_a"] = emit(cut(NEW_, P_AB, DUR), "y31_new_a", "clip new")

    # y2：补不补噪声细节（第1段=现役，第2段=现役+细节）
    made["y31_new_d0"] = emit(cut(NEW_, P_AB, DUR), "y31_new_d0", "clip d0")
    made["y31_new_d1"] = emit(with_detail(cut(NEW_, P_AB, DUR), "d1"),
                              "y31_new_d1", "clip d1")

    # y3：基线题
    made["y31_new_b"] = emit(cut(NEW_, P_BASE, DUR), "y31_new_b", "clip base")
    made["y31_el_b"] = emit(cut(EL_S, P_BASE, DUR), "y31_el_b", "clip el")

    ctrl = cut(NEW_, P_BASE, DUR)
    made["y31_ctrl1"] = emit(ctrl, "y31_ctrl1", "ctrl A")
    made["y31_ctrl2"] = emit(ctrl, "y31_ctrl2", "ctrl B")

    print("\n=== md5 自检 ===")
    inv = {}
    for nm, p in made.items():
        inv.setdefault(md5(p), []).append(nm)
    bad = False
    for h, ns in inv.items():
        if len(ns) > 1:
            ok = set(ns) <= {"y31_ctrl1", "y31_ctrl2"}
            print("  %-9s %s" % (h, "OK(控制题同素材)" if ok
                                 else "🔴 撞车 -> %s" % ns))
            if not ok:
                bad = True
    if not bad:
        print("  ✅ 无撞车")

    items = [
        # ---- 1. 核心：融合补丁有没有修掉三症状 ----
        {"id": "y1_symptom_ab_20261015y", "dim": "电子音/吃字/卡字（新旧对比）",
         "title": "这两段，哪一段的「电子音 / 吃字 / 卡字」更少？",
         "desc": "两段都是男声、同一段文案，各 30 秒（都做了响度对齐）。\n\n"
                 "**第 1 段和第 2 段都是我们的版本**，只是处理方式不同。\n"
                 "🔴 我们改的是：以前同一段语音被时域拉伸算法**揉了两遍**，"
                 "现在合并成**一遍**。\n"
                 "请重点听：**字的发音糊不糊、有没有卡住、声音假不假。**",
         "ref": None,
         "note": "🔴🔴 本轮核心题。\n"
                 "第1段=旧版（变速 atempo + 变调 atempo，两步几乎互相抵消，白揉两遍）\n"
                 "第2段=新版（合并成一次，实测 atempo=1.007，几乎不再拉伸）\n\n"
                 "机器侧（494s 同文案，同口径）：\n"
                 "  Σz²：72.2 → 27.1（EL 自己中位 59.6，已进入 EL 分布）\n"
                 "  平坦度：0.0242 → 0.0373（EL 0.0461）\n"
                 "  150~300Hz：+1.80σ → −0.41σ\n\n"
                 "选①第1段更少 → 融合反而有害，回退\n"
                 "选②第2段更少 → ✅ 修法被证实，定案\n"
                 "选③差不多   → 三症状另有主因，下轮查停顿合成/拼接点\n"
                 "选④两段都明显 → 同上，且需换定位素材（可能这段听不出）",
         "options": [
             {"key": "A", "file": "y31_old_a.mp3", "label": "① 第 1 段更少"},
             {"key": "B", "file": "y31_new_a.mp3", "label": "② 第 2 段更少"},
             {"key": "C", "file": None, "label": "③ 两段差不多，听不出区别"},
             {"key": "D", "file": None, "label": "④ 两段都很明显"}]},

        # ---- 2. 补不补「噪声细节」 ----
        {"id": "y2_detail_ab_20261015y", "dim": "电子音：补不补噪声细节",
         "title": "这两段，哪一段的「电子感」更少？",
         "desc": "两段都是男声、同一段文案，各 30 秒（都做了响度对齐）。\n\n"
                 "**第 1 段和第 2 段都是当前线上版本**，第 2 段额外补了一层"
                 "很轻的噪声细节（真实语音里送气/摩擦声那部分）。\n"
                 "请只判断**电子感**，不要管别的。",
         "ref": None,
         "note": "🔴 动作明确：\n"
                 "  选② → 上线（设 TTS_DETAIL_K=0.04，带限 1.5k~4k）\n"
                 "  选① → 反而变差，不上线\n"
                 "  选③ → 听不出，不上线（省一次编码风险）\n"
                 "  选④ → 两段都电子，说明电子音不在这一维，转查音色本身\n\n"
                 "机器侧（494s 全篇口径）：\n"
                 "  现役平坦度 0.03867，EL 0.04329 → 只差 1.12 倍（已很接近）\n"
                 "  ⚠️ 但现役 4k~8k 已 +3.13σ（偏多），补噪声会把它推更高：\n"
                 "     宽带粉噪 k=0.06 → 4k~8k 1.838（EL 1.312，超 40%）\n"
                 "     带限1.5k-4k k=0.04 → 4k~8k 1.594（超 21%），平坦度 0.0408\n"
                 "  → 所以这轮用的是**带限**版本，k 也压到 0.04。\n"
                 "  🔴 风险：补噪声可能听成「沙沙声」，必须让员工裁决。",
         "options": [
             {"key": "A", "file": "y31_new_d0.mp3", "label": "① 第 1 段（现役）电子感更少"},
             {"key": "B", "file": "y31_new_d1.mp3", "label": "② 第 2 段（补了细节）电子感更少"},
             {"key": "C", "file": None, "label": "③ 两段差不多，听不出区别"},
             {"key": "D", "file": None, "label": "④ 两段都很电子"}]},

        # ---- 3. 长期基线锚点 ----
        {"id": "y3_send_baseline_20261015y", "dim": "可发送性",
         "title": "这 30 秒，哪一段可以直接发给客户？",
         "desc": "按「能不能直接发给客户」判断。\n\n"
                 "第 1 段是我们**刚改过的版本**（语音不再被拉伸算法揉两遍），"
                 "第 2 段是 EL。",
         "ref": None,
         "note": "**长期基线锚点。**\n"
                 "此前 4 轮共 **15 张票无一例外全选 EL**。\n"
                 "本轮是我们 Σz² 首次进入 EL 分布（27.1 < EL 中位 59.6）之后的第一票。\n"
                 "→ 哪天出现「我们也能发」，才算真有进展。",
         "options": [
             {"key": "A", "file": "y31_new_b.mp3", "label": "第 1 段（我们的）"},
             {"key": "B", "file": "y31_el_b.mp3", "label": "第 2 段（EL）"},
             {"key": "C", "file": None, "label": "都不行"}]},
    ]

    items.append(
        {"id": "ctrl_same_20261015y", "dim": "答题有效性校验",
         "title": "【最后一题】这两段，听起来有区别吗？",
         "desc": "**这道题必须选一个**（选错也没关系）。\n\n"
                 "这两段是**同一段音频放了两遍**，内容完全一样。\n"
                 "听 5 秒就够。",
         "ref": None,
         "note": "隐藏控制题：正解=C 两段一样。\n"
                 "🔴 沿用问法「这两段听起来有区别吗」——已连续 3 轮全对；\n"
                 "   旧问法「哪段念得更快」4 人里 2 人错，别再用。",
         "options": [
             {"key": "A", "file": "y31_ctrl1.mp3", "label": "第 1 段不一样"},
             {"key": "B", "file": "y31_ctrl2.mp3", "label": "第 2 段不一样"},
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
        shutil.copyfile(p, p + ".bak-before-y")
        print("\n已备份旧题库 ->", p + ".bak-before-y")
    json.dump({"items": items}, open(p, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("题库已写入 %s  共 %d 题" % (p, len(items)))
    if bad or empty:
        print("🔴 存在拦截项，先别让员工答题")
    print("\n发布后确认：curl -s http://服务器:8000/api/fb/where")
    print("归档：python3 /mnt/tts/diag/fb_loop.py --archive x")


if __name__ == "__main__":
    sys.exit(main() or 0)
