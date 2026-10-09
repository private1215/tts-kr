#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""韵律五项客观指标提取器 —— 把"语调/停顿/重音/呼吸/情绪"变成可算的数。

## 为什么必须写它

人工盲听给出的是"感觉哪里不对"，但"感觉"没法回归、没法定位、没法验收。
这五项里至少四项**是可以客观测量的**，依据是两条外部标准：

1. **K-ToBI**（Sun-Ah Jun, UCLA）—— 韩语韵律标注国际标准。
   核心单位：AP（Accentual Phrase 声调短语）/ IP（Intonation Phrase 语调短语）。
   - IP 末尾由**边界音调**标记：L% H% LH% HL% LHL% HLH% LHLH% HLHL% LHLHL% 共 8 类
   - 其中 **HL% 在新闻广播中最常见**（Jun 标注手册原文："commonly used in news
     broadcasting"）—— 本项目正是财经广播，这就是我们的目标分布锚点
   - 韩语是 **edge-prominence 语言**：焦点落在短语**左边缘**而非音节重音
   - AP 首音节音调由首辅音喉部特征决定：送气/紧音开头 → H，否则 → L
   - 标注者一致性：IP 边界 91%、AP 边界 78%、break index ±1 内 99%
     → 说明这套体系是稳定可复现的，不是玄学

2. **INTERSPEECH 2024**（Yang/Koriyama/Saito, arXiv:2402.00288）—— 呼吸检测。
   用 duration + ZCR + mel 谱方差(VMS) + NA-VMS 四个声学特征，就能把
   **呼吸声从纯静音里分出来**，无需训练模型（IoU 0.836 / precision 0.924）。
   该论文同时证明：带呼吸的 TTS 自然度 MOS 3.55 > 不带 3.34。

另有 울산대 研究：韩语 TTS 只需把韵律边界简化成"上升/平坦/下降"三类，
MOS 就能显著提升 —— 所以**不需要完整 K-ToBI，三类句尾音调就够用**。

## 五项各自的口径

| 项 | 指标 | 依据 |
|---|---|---|
| 语调 | 句尾边界音调三类占比 + 升降幅度(semitone) | K-ToBI IP boundary tone |
| 停顿 | 密度(次/分) + 时长分位 + 长短结构 | K-ToBI break index |
| 重音 | 句内 F0 峰值相对位置 + 突显度 | 韩语 edge-prominence |
| 呼吸 | 频率(次/分) + 时长 + 占静音比 | INTERSPEECH2024 声学特征 |
| 情绪 | 句间音高/能量变异系数 + 句内动态 | arousal 声学代理 |

⚠️ 边界：情绪是**最弱**的一项 —— 上面只是声学唤起度代理，不等于"情绪对不对"。
它只能告诉你"起伏够不够"，不能告诉你"该高兴的地方高兴了没有"。

## 用法（服务器，必须用 qwen venv：librosa 1.0.0）

    venv/bin/python tools/prosody_target.py --limit 6            # 先试跑
    venv/bin/python tools/prosody_target.py --out /mnt/tts/diag/prosody_target.json
    venv/bin/python tools/prosody_target.py --wav SOME.wav       # 量单个产物
"""
import os
import re
import sys
import json
import glob
import argparse

import numpy as np
import librosa
import soundfile as sf

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from app.prosody_report import load, f0_of  # noqa: E402

CASES_DIR = os.environ.get("TTS_CASES_DIR", "/mnt/tts/data/cases")
CASES_JSON = os.environ.get("TTS_CASES_JSON", "/mnt/tts/data/cases.json")

SR = 24000

# ---- 静音检测 ----
# 🔴 第一版用 min_sil=0.10 + pct(db,30)+8，跑出"停顿 107 次/分钟"（每 0.56s 一次），
#    那是把音节间的微小间隙也算成停顿了，切出来的"句子"其实是语音碎片，
#    导致句尾语调分类变成随机（fall 0.42 / rise 0.42 对称分布 = 典型乱判）。
#    改成更严的口径：pct(db,25)+6（与 prosody_report.sent_windows 同源），
#    并且"停顿"和"句边界"用**两级**阈值，句边界要求更长的静音。
SIL_MIN = 0.18       # 算作停顿的最小静音
SENT_MIN = 0.35      # 算作句子边界的最小静音（切句子用，语调分类只在真句尾做）
# ---- 句尾边界音调分类阈值（semitone）----
# 人耳对音高变化的察觉阈值约 1 semitone，低于此判 rise/fall 等于放大噪声
RISE_ST = 1.0
FALL_ST = -1.0
# ---- 呼吸判定阈值（数据驱动，见 _breath_feats）----
BREATH_MIN_DUR = 0.06
BREATH_MAX_DUR = 1.20
# 🔴 绝对能量下限：光看"高出底噪 6dB"不够。生成产物的静音段很干净，
#    会把 -20.9dB 的极弱底噪也算成呼吸（实测误判成 4.6 次/分）。
#    而案例真实静音段的 rel_db 中位是 -11.2dB。低于 -16dB 的一律不算呼吸。
BREATH_MIN_REL_DB = float(os.environ.get("TTS_BREATH_MIN_REL_DB", "-16.0"))


# ---------------------------------------------------------------- 静音/句子

def sil_gaps(x, sr, min_sil=SIL_MIN):
    """低能量区间（静音+呼吸都落在里面）。返回 [(start,end), ...] 单位为采样点。"""
    hop, win = int(0.01 * sr), int(0.03 * sr)
    n = max((len(x) - win) // hop, 1)
    rms = np.sqrt(np.array([np.mean(x[i * hop:i * hop + win] ** 2)
                            for i in range(n)]) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    thr = np.percentile(db, 25) + 6.0
    sil = db < thr
    gaps, i = [], 0
    while i < n:
        if sil[i]:
            j = i
            while j < n and sil[j]:
                j += 1
            if (j - i) * hop / sr >= min_sil:
                gaps.append((i * hop, min(j * hop, len(x))))
            i = j
        else:
            i += 1
    return gaps, db, hop


def sent_segs(x, sr, gaps, min_len=0.4, max_len=25.0):
    """用**句边界级**的静音（≥SENT_MIN）切句子。

    🔴 不能用全部停顿来切句：短停顿（0.18~0.35s）只是短语内换气，
    拿它的尾端当"句尾"去做语调分类，量到的是句中音高，不是边界音调。
    """
    big = [(a, b) for a, b in gaps if (b - a) / sr >= SENT_MIN]
    bnd = [0]
    for a, b in big:
        bnd += [a, b]
    bnd.append(len(x))
    segs = []
    for k in range(0, len(bnd) - 1, 2):
        s, e = bnd[k], bnd[k + 1]
        if min_len <= (e - s) / sr <= max_len:
            segs.append((s, e))
    return segs


# ---------------------------------------------------------------- 语调

def boundary_tone(f):
    """句尾边界音调分类（简化 K-ToBI 三类）。

    取句子最后 40% 的 F0，换算成相对本句中位数的半音，
    用"净变化量 + 线性斜率"双判据，避免单帧抖动误判。
    """
    n = len(f)
    # 取最后 35%，并掐掉最末 5% —— 句末拖尾常有气声，pyin 在那段的 F0 不可靠
    tail = f[int(n * 0.60):max(int(n * 0.95), int(n * 0.60) + 6)]
    if len(tail) < 6:
        return None, None
    med = float(np.median(f))
    st = 12.0 * np.log2(tail / max(med, 1e-6) + 1e-12)
    # 3 点滑动平均：单帧抖动会让端点差乱跳
    k = 3
    if len(st) >= k:
        st = np.convolve(st, np.ones(k) / k, mode="valid")
    d = float(st[-1] - st[0])
    slope = float(np.polyfit(np.arange(len(st)), st, 1)[0])
    if d > RISE_ST and slope > 0:
        return "rise", d
    if d < FALL_ST and slope < 0:
        return "fall", d
    return "flat", d


# ---------------------------------------------------------------- 重音

def prominence(f, f_times):
    """句内 F0 峰值的相对位置 + 突显度。韩语是 edge-prominence，
    峰值应偏左；峰值位置分布本身就是重音风格的指纹。"""
    if len(f) < 8:
        return None, None
    med = float(np.median(f))
    st = 12.0 * np.log2(f / max(med, 1e-6) + 1e-12)
    k = int(np.argmax(st))
    pos = k / float(len(st) - 1)
    amp = float(np.percentile(st, 90) - np.percentile(st, 10))
    return pos, amp


# ---------------------------------------------------------------- 呼吸

def breath_feats(x, sr, gaps):
    """从静音区间里把"呼吸"分出来（INTERSPEECH2024 思路，纯规则不需要训练）。

    呼吸 = 有气流摩擦的宽带噪声：过零率高、频谱平坦、但能量明显低于语音。
    纯静音 = 能量极低且频谱接近底噪。
    """
    out = []
    voiced_ref = np.percentile(np.abs(x[np.abs(x) > 1e-5]) if np.any(np.abs(x) > 1e-5)
                               else np.abs(x), 75)
    for a, b in gaps:
        dur = (b - a) / sr
        if not (BREATH_MIN_DUR <= dur <= BREATH_MAX_DUR):
            continue
        seg = x[a:b]
        if len(seg) < 32:
            continue
        rms = float(np.sqrt(np.mean(seg ** 2) + 1e-12))
        # 过零率
        zcr = float(np.mean(np.abs(np.diff(np.sign(seg)))) / 2.0)
        # 频谱平坦度：噪声(呼吸)高，纯静音/底噪低
        S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        S = np.clip(S, 1e-12, None)
        flat = float(np.exp(np.mean(np.log(S))) / (np.mean(S) + 1e-12))
        # 相对语音的能量(dB)
        rel_db = 20 * np.log10(rms / max(voiced_ref, 1e-12) + 1e-12)
        # 高频占比：呼吸的摩擦噪声高频成分比静音底噪多
        hf = float(np.sum(S[len(S) // 2:]) / (np.sum(S) + 1e-12))
        out.append({"dur": dur, "rms": rms, "zcr": zcr, "flat": flat,
                    "rel_db": rel_db, "hf": hf, "at": round(a / sr, 2)})
    return out


def pick_breath(feats):
    """数据驱动：静音段里 rel_db 明显高于最低档、且频谱较平坦的算呼吸。
    阈值不拍脑袋 —— 先按 rel_db 排序取分位，见 main 里的自适应分位。"""
    if not feats:
        return [], []
    rel = np.array([f["rel_db"] for f in feats])
    if len(rel) < 6:
        return [], feats
    # 纯静音 = 能量最低的一档；用它当基线，高出 6dB 以上且频谱平坦 → 呼吸
    base = float(np.percentile(rel, 10))
    fl = np.array([f["flat"] for f in feats])
    thr_f = float(np.percentile(fl, 40))
    br = [f for f in feats
          if f["rel_db"] > base + 6.0
          and f["rel_db"] > BREATH_MIN_REL_DB
          and f["flat"] >= thr_f]
    return br, feats


# ---------------------------------------------------------------- 主测量

def measure_wav(path, max_segs=40):
    x, sr = load(path, sr0=SR)
    total = len(x) / sr
    gaps, db, hop = sil_gaps(x, sr)
    segs = sent_segs(x, sr, gaps)
    if len(segs) > max_segs:
        # 均匀抽样，避免只测开头
        idx = np.linspace(0, len(segs) - 1, max_segs).astype(int)
        segs = [segs[i] for i in sorted(set(idx.tolist()))]

    # --- 语调 / 重音 ---
    tones, amps, poss, promps, sentp, sente = [], [], [], [], [], []
    for s, e in segs:
        f = f0_of(x[s:e], sr)
        if f is None:
            continue
        t, d = boundary_tone(f)
        if t:
            tones.append(t)
            amps.append(d)
        p, amp = prominence(f, None)
        if p is not None:
            poss.append(p)
            promps.append(amp)
        sentp.append(float(np.median(f)))
        seg = x[s:e]
        sente.append(float(np.sqrt(np.mean(seg ** 2) + 1e-12)))

    # --- 停顿 ---
    pdur = np.array([(b - a) / sr for a, b in gaps]) if gaps else np.array([0.0])

    # --- 呼吸 ---
    feats = breath_feats(x, sr, gaps)
    br, allg = pick_breath(feats)

    def cv(v):
        v = np.asarray(v, dtype=float)
        return float(np.std(v) / (np.mean(v) + 1e-12)) if len(v) > 2 else None

    def q(v, p):
        v = np.asarray(v, dtype=float)
        return float(np.percentile(v, p)) if len(v) else None

    n_t = float(len(tones)) if tones else 1.0
    return {
        "wav": os.path.basename(path),
        "dur": round(total, 1),
        "segs": len(segs),
        # 语调
        "tone_fall": round(tones.count("fall") / n_t, 3),
        "tone_flat": round(tones.count("flat") / n_t, 3),
        "tone_rise": round(tones.count("rise") / n_t, 3),
        "tone_amp_p50": round(q(amps, 50), 2) if amps else None,
        "tone_samples": len(tones),
        # 停顿
        "pause_n_per_min": round(len(gaps) / (total / 60.0), 1),
        "pause_dur_p10": round(q(pdur, 10), 3),
        "pause_dur_p50": round(q(pdur, 50), 3),
        "pause_dur_p90": round(q(pdur, 90), 3),
        "pause_long_share": round(float(np.mean(pdur >= 0.5)), 3) if len(pdur) else None,
        # 重音（峰值位置 0=句首 1=句末）
        "prom_pos_p50": round(q(poss, 50), 3),
        "prom_pos_p25": round(q(poss, 25), 3),
        "prom_amp_p50": round(q(promps, 50), 2),
        # 呼吸
        "breath_n_per_min": round(len(br) / (total / 60.0), 1),
        "breath_dur_p50": round(q([f["dur"] for f in br], 50), 3) if br else None,
        "breath_share_of_gaps": round(len(br) / max(len(allg), 1), 3),
        "breath_rel_db_p50": round(q([f["rel_db"] for f in br], 50), 1) if br else None,
        "gap_samples": len(allg),
        # 情绪（唤起度代理）
        "sent_pitch_cv": cv(sentp),
        "sent_energy_cv": cv(sente),
        "pitch_med": round(q(sentp, 50), 1),
    }


def merge(items):
    keys = ["tone_fall", "tone_flat", "tone_rise", "tone_amp_p50",
            "pause_n_per_min", "pause_dur_p50", "pause_dur_p90", "pause_long_share",
            "prom_pos_p50", "prom_amp_p50",
            "breath_n_per_min", "breath_dur_p50", "breath_share_of_gaps",
            "sent_pitch_cv", "sent_energy_cv"]
    out = {}
    for k in keys:
        v = [i[k] for i in items if i.get(k) is not None]
        if v:
            out[k] = {"p25": round(float(np.percentile(v, 25)), 3),
                      "p50": round(float(np.percentile(v, 50)), 3),
                      "p75": round(float(np.percentile(v, 75)), 3),
                      "n": len(v)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条案例（试跑用）")
    ap.add_argument("--gender", default="", help="只跑某性别")
    ap.add_argument("--max-segs", type=int, default=40)
    ap.add_argument("--wav", default="", help="单独量一个 wav（生成产物对比用）")
    ap.add_argument("--out", default="/mnt/tts/diag/prosody_target.json")
    a = ap.parse_args()

    if a.wav:
        print(json.dumps(measure_wav(a.wav, a.max_segs), ensure_ascii=False, indent=2))
        return

    cases = json.load(open(CASES_JSON, encoding="utf-8"))
    cases = cases if isinstance(cases, list) else list(cases.get("cases", cases.values()))
    if a.gender:
        cases = [c for c in cases if c.get("gender") == a.gender]
    if a.limit:
        cases = cases[:a.limit]

    rows = []
    for c in cases:
        w = os.path.join(CASES_DIR, c["id"], "merged.wav")
        if not os.path.exists(w):
            continue
        try:
            m = measure_wav(w, a.max_segs)
        except Exception as e:
            print("跳过 %s: %s" % (c["id"], repr(e)[:120]), flush=True)
            continue
        m["id"] = c["id"]
        m["gender"] = c.get("gender")
        m["name"] = c.get("name")
        rows.append(m)
        print("OK %s %s  fall=%.2f flat=%.2f rise=%.2f  停顿/分=%.1f  呼吸/分=%.1f"
              % (c["id"], c.get("gender"), m["tone_fall"], m["tone_flat"],
                 m["tone_rise"], m["pause_n_per_min"], m["breath_n_per_min"]),
              flush=True)

    res = {
        "n_cases": len(rows),
        "all": merge(rows),
        "male": merge([r for r in rows if r.get("gender") == "male"]),
        "female": merge([r for r in rows if r.get("gender") == "female"]),
        "per_case": rows,
    }
    json.dump(res, open(a.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n===== 汇总（中位数 p50）=====")
    for grp in ["male", "female", "all"]:
        if not res[grp]:
            continue
        print("\n[%s] n=%d" % (grp, len([r for r in rows
              if grp == "all" or r.get("gender") == grp])))
        for k, v in res[grp].items():
            print("  %-22s p50=%-8s (p25=%s p75=%s)" % (k, v["p50"], v["p25"], v["p75"]))
    print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
