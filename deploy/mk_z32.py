#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第三十二轮（z）题库：**音色 vs 音高的工作点裁决**

## 为什么换题（y 轮作废）
y 轮素材是 ，而线上已换成 h5 工作点 → 违反
「定位类题给员工听的必须是当前线上原状」，y 轮据此作废重出。

## 这一轮改了什么
发现「音色相似度（campplus，对 EL_male.wav）」上：
  裸推理 0.8530 → **变调步 −0.1770 → 0.6739**（占全部损失 62%）
机理： 整体重采样把 **F0 和共振峰一起**搬 R 倍，共振峰 = 说话人身份载体。
生产 R=1.094（为把 180.8Hz 拉到 197.2Hz）→ 音色 0.666，**低于同人阈值 0.75**。

### 于是把工作点换成 h5（音色优先）
| | Σz² | 音色 | F0 |
|---|---|---|---|
| 旧 r35（目标 F0 197.7） | 10.6 | 0.6841 | 194.9 |
| **新 h5（目标 188 + HP110 + 250Hz+1.2）** | **33.0** | **0.7986** | 182.9 |
EL 自身 Σz² 分布 17~240（中位 59.6）→ h5 的 33.0 才叫「落进分布」。
🔴 不能简单「不变调」：60~150Hz 会涨到 EL 的 **2 倍（+9.05σ，就是「闷」）**。

## 本轮 4 题 + 控制题
| 题 | 答案 → 动作 |
|---|---|
| z1 音色优先 vs 音高优先 | 选音色 → h5 定案；选音高 → 回退 r35 |
| z2 能否直接发客户 | 长期锚点（此前 15 票全选 EL）|
| z3 三症状还在不在（新素材） | ③没有 → 该项关闭 |
| z4 补不补噪声细节 | ②→开 TTS_DETAIL_K=0.04；①/③→不开 |
"""
import hashlib, json, os, shutil, subprocess, sys

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")
import detail  # noqa: E402

FB_DIR = os.environ.get("TTS_FB_DIR", "/mnt/tts/fb")
TMPD = "/mnt/tts/diag/_z32"
TGT_LUFS = -26.1
DUR = 30.0
SR_CLIP = 24000

NEW_ = "/mnt/tts/output/longprobe-z32-6cdd/longprobe-z32-6cdd-1.mp3"
OLD_ = "/mnt/tts/output/longprobe-r35-6cdd/longprobe-r35-6cdd-1.mp3"
EL_S = "/mnt/tts/data/cases/6cdd25bc75f2/merged.wav"

P_AB = 210.0
P_BASE = 250.0
DETAIL_K = 0.04


def lufs(p):
    r = subprocess.run(["ffmpeg", "-i", p, "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    v = [l for l in r.stderr.splitlines() if "I:" in l and "LUFS" in l]
    return float(v[-1].split("I:")[1].split("LUFS")[0].strip()) if v else None


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()[:8]


def cut(src, start, dur):
    t = os.path.join(TMPD, "_t%s_%d.wav" % (os.path.basename(src)[:12], int(start * 1000)))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-ss", "%.1f" % start, "-ar", str(SR_CLIP), "-ac", "1",
                    "-t", "%.1f" % dur, t], check=True)
    got = os.path.getsize(t) / 2.0 / SR_CLIP
    if got < dur - 0.5:
        sys.exit("🔴 切片越界：%s 第%.0fs 起取%.0fs，实际仅%.1fs"
                 % (os.path.basename(src), start, dur, got))
    return t


def emit(wav, name, title):
    g = TGT_LUFS - lufs(wav)
    out = os.path.join(FB_DIR, "%s.mp3" % name)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav,
                    "-af", "volume=%gdB" % g, "-b:a", "192k",
                    "-metadata", "title=%s" % title, out], check=True)
    back = lufs(out)
    print("  %-18s md5=%s  LUFS %6.1f %s"
          % (name, md5(out), back if back else 0,
             "OK" if back and abs(back - TGT_LUFS) < 1.0 else "!!爆表"))
    return out


def with_detail(wav, name):
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

    # z1：音色优先 vs 音高优先（A=旧 r35，B=新 h5）
    made["z32_old_a"] = emit(cut(OLD_, P_AB, DUR), "z32_old_a", "clip old")
    made["z32_new_a"] = emit(cut(NEW_, P_AB, DUR), "z32_new_a", "clip new")

    # z2：基线题（A=新 h5，B=EL）
    made["z32_new_b"] = emit(cut(NEW_, P_BASE, DUR), "z32_new_b", "clip base")
    made["z32_el_b"] = emit(cut(EL_S, P_BASE, DUR), "z32_el_b", "clip el")

    # z3：三症状复核（单条 B=新 h5）
    made["z32_new_c"] = emit(cut(NEW_, P_AB, DUR), "z32_new_c", "clip sym")

    # z4：补不补噪声细节（A=新 h5，B=新 h5+细节）
    made["z32_new_d0"] = emit(cut(NEW_, P_AB, DUR), "z32_new_d0", "clip d0")
    made["z32_new_d1"] = emit(with_detail(cut(NEW_, P_AB, DUR), "d1"),
                               "z32_new_d1", "clip d1")

    # 控制题：同一素材两遍
    ctrl = cut(NEW_, P_BASE, DUR)
    made["z32_ctrl1"] = emit(ctrl, "z32_ctrl1", "ctrl A")
    made["z32_ctrl2"] = emit(ctrl, "z32_ctrl2", "ctrl B")

    print("\n=== md5 自检 ===")
    inv = {}
    for nm, p in made.items():
        inv.setdefault(md5(p), []).append(nm)
    bad = False
    for h, ns in inv.items():
        if len(ns) > 1:
            ok = set(ns) <= {"z32_ctrl1", "z32_ctrl2"}
            print("  %-9s %s" % (h, "OK(控制题同素材)" if ok else "🔴 撞车 -> %s" % ns))
            if not ok:
                bad = True
    if not bad:
        print("  ✅ 无撞车")

    items = [
        {"id": "z1_workpoint_ab_20261016z", "dim": "音色 vs 音高（工作点裁决）",
         "title": "这两段，哪一段听起来更像是「同一个人在自然说话」？",
         "desc": "两段都是男声、同一段文案、各 30 秒（响度已对齐）。\n\n"
                 "**两段都是我们的版本**，区别只有一处：\n"
                 "· 第 1 段 —— 整体音高更接近 EL\n"
                 "· 第 2 段 —— 声音质感（音色）更接近 EL\n\n"
                 "请整体判断：**哪一段更像「同一个人在正常、自然地说话」？**\n"
                 "不要只听音高低，也别只听某一个字。",
         "ref": None,
         "note": "🔴🔴 本轮核心裁决题。\n"
                 "第1段 = 旧工作点 r35（目标 F0 197.7，需 R=1.094 上搬）\n"
                 "第2段 = 新工作点 h5（目标 F0 188 + HP110 + 250Hz+1.2）\n\n"
                 "机器侧（同素材同长度，campplus 对 EL_male.wav，同人阈值 0.75）：\n"
                 "  旧 r35：音色 0.6841（**低于同人阈值**）、Σz² 10.6、F0 194.9Hz\n"
                 "  新 h5 ：音色 0.7986（过阈值）、Σz² 33.0、F0 182.9Hz\n"
                 "  上限（EL 自己）：0.9628；EL 自身 Σz² 分布 17~240（中位 59.6）\n"
                 "  机理：asetrate 整体重采样把 F0 与共振峰一起搬，生产搬 9.4%。\n\n"
                 "选① 更接近 EL 音高 → 回退 r35\n"
                 "选② 更像同一个人   → ✅ h5 定案，继续沿「音色优先」走\n"
                 "选③ 差不多         → 说明这一维听不出来，改去攻低频/闷\n"
                 "选④ 都不自然       → 两段都不行，主因不在这一维",
         "options": [
             {"key": "A", "file": "z32_old_a.mp3", "label": "① 第 1 段（音高更接近 EL）"},
             {"key": "B", "file": "z32_new_a.mp3", "label": "② 第 2 段（音色更接近 EL）"},
             {"key": "C", "file": None, "label": "③ 两段差不多，听不出区别"},
             {"key": "D", "file": None, "label": "④ 两段都不自然"}]},

        {"id": "z3_symptom_20261016z", "dim": "电子音/吃字/卡字（当前线上）",
         "title": "这一段里，还有没有「电子音 / 吃字 / 卡字」？",
         "desc": "这是我们的**当前线上版本**，男声 30 秒。\n\n"
                 "请重点听三件事：\n"
                 "· **电子音**：声音发假、像机器合成\n"
                 "· **吃字**：某个字的辅音被吞掉、听不清\n"
                 "· **卡字**：字与字之间卡顿、突然中断\n\n"
                 "按最明显的那一项判断。",
         "ref": None,
         "note": "🔴 上一个版本（揉两遍 WSOLA）已被定位为三症状的根因并修掉，\n"
                 "本轮素材是修复+换工作点之后的当前线上原状。\n\n"
                 "选③ 没有 → 三项从问题清单关闭\n"
                 "选② 轻微 → 降级，但仍留一个长期观察项\n"
                 "选① 明显 → 根因没找全，回查停顿合成/拼接点/后处理顺序",
         "options": [
             {"key": "A", "file": "z32_new_c.mp3", "label": "① 明显还有"},
             {"key": "B", "file": None, "label": "② 有，但很轻微"},
             {"key": "C", "file": None, "label": "③ 没有"}]},

        {"id": "z2_send_baseline_20261016z", "dim": "可发送性",
         "title": "这 30 秒，哪一段可以直接发给客户？",
         "desc": "按「能不能直接发给客户」判断。\n\n"
                 "第 1 段是我们的当前线上版本，第 2 段是 EL。",
         "ref": None,
         "note": "**长期基线锚点。**\n"
                 "此前 4 轮共 15 张票无一例外全选 EL。\n"
                 "本轮是音色首次跨过「同人阈值 0.75」（0.684 → 0.799）之后的第一票。\n"
                 "→ 哪天出现「我们也能发」，才算真有进展。",
         "options": [
             {"key": "A", "file": "z32_new_b.mp3", "label": "第 1 段（我们的）"},
             {"key": "B", "file": "z32_el_b.mp3", "label": "第 2 段（EL）"},
             {"key": "C", "file": None, "label": "都不行"}]},

        {"id": "z4_detail_ab_20261016z", "dim": "电子音：补不补噪声细节",
         "title": "这两段，哪一段的「电子感」更少？",
         "desc": "两段都是当前线上版本、同一段文案、各 30 秒（响度已对齐）。\n\n"
                 "第 2 段额外补了一层很轻的噪声细节（真实语音里送气/摩擦声那部分）。\n"
                 "请只判断**电子感**，不要管别的。",
         "ref": None,
         "note": "🔴 动作明确：\n"
                 "  选② → 上线（TTS_DETAIL_K=0.04，带限 1.5k~4k）\n"
                 "  选① → 反而变差，不上线\n"
                 "  选③ → 听不出，不上线（省一次编码风险）\n"
                 "  选④ → 电子音不在这一维，转查音色/停顿\n\n"
                 "机器侧：补噪声会推高 4k~8k（带限 k=0.04 → 超 EL 21%）→ 收益<风险，只能 A/B 裁决。",
         "options": [
             {"key": "A", "file": "z32_new_d0.mp3", "label": "① 第 1 段（现役）电子感更少"},
             {"key": "B", "file": "z32_new_d1.mp3", "label": "② 第 2 段（补了细节）电子感更少"},
             {"key": "C", "file": None, "label": "③ 两段差不多，听不出区别"},
             {"key": "D", "file": None, "label": "④ 两段都很电子"}]},
    ]

    items.append(
        {"id": "ctrl_same_20261016z", "dim": "答题有效性校验",
         "title": "【最后一题】这两段，听起来有区别吗？",
         "desc": "**这道题必须选一个**（选错也没关系）。\n\n"
                 "这两段是**同一段音频放了两遍**，内容完全一样。听 5 秒就够。",
         "ref": None,
         "note": "隐藏控制题：正解=C 两段一样。\n"
                 "🔴 沿用问法「这两段听起来有区别吗」——已连续 3 轮全对；\n"
                 "   旧问法「哪段念得更快」4 人里 2 人错，别再用。",
         "options": [
             {"key": "A", "file": "z32_ctrl1.mp3", "label": "A 明显有区别"},
             {"key": "B", "file": "z32_ctrl2.mp3", "label": "B 明显有区别"},
             {"key": "C", "file": None, "label": "C 没有区别，是一样的"}]})

    out = os.path.join(FB_DIR, "items.json")
    shutil.copyfile(out, os.path.join(FB_DIR, "items_roundy.json"))
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"items": items}, f, ensure_ascii=False, indent=1)
    print("\n发布 %d 题 / %d 段音频 -> %s"
          % (len(items), len(made), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
