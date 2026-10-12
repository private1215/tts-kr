# -*- coding: utf-8 -*-
"""同文本对照的「语调 / 情绪」量化对比 —— 回答「只测音色够不够」。

背景（2026-10-12）：
    GPT-SoVITS 微调把音色相似度做到 0.9426（女）/ 0.9521（男），
    但**音色不等于听感**。员工人工盲听反馈的是「语调不对、断句不对、
    像念稿」—— 这些全在韵律维度，campplus 相似度一个数都看不见。
    本脚本就是补这一块：同文本、同参照（案例未训练区原声），把
    语调/情绪相关的可量化维度全部摆出来对比。

测什么（都是「听得出差别」才收进来的）：
    med      整段浊音帧 F0 中位(Hz)      —— 音高基线对不对
    A        句内音高动态(semi)           —— 有没有声调起伏
    B        全局音高跨度(semi)           —— 用了多少音域
    C        能量动态(dB)                 —— 有没有轻重音
    D        语速变化 CV                  —— 语速是不是一路到底
    E        句尾斜率标准差               —— 尾声是不是一个调
    F        停顿熵(0~1)                  —— 停顿是不是只有一个长度
    drop     句尾定向降幅(semi, 负=降调)  —— 句尾"落地"了没有
    sps      语速(音节/秒)                —— 快了还是慢了
    prate    停顿密度(次/分钟)            —— 断句节奏

⚠️ 三条硬约束（违反了分数就没有意义）：
    1. **必须同文本**。不同文案的停顿数、时长、音节数天然不同，
       跨文案比 sps / prate / F 是拿噪声当结论。
    2. **必须同口径**。F0 一律走 Praat（app/proso.tier_of），
       不能用 pyin 混着量 —— pyin 与 Praat 数值不同源（见 proso 注释陷阱 #4）。
    3. **性别必须由案例库标注给**，不要用 F0 反推
       （pyin 八度错误已两次翻车，见 MEMORY.md）。

用法：
    python tools/prosody_cmp.py --spec 女声.json
    # spec 是 JSON: [{"name":"...", "path":"...", "is_male":false, "text":"..."}...]
    # 第一条必须是参照（案例原声），后面的都跟它比。
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import prosody_report as PR                       # noqa: E402
from app import proso                                      # noqa: E402

TAIL_SEC = 0.35


def _to_wav(path):
    """mp3/m4a → 24k mono wav 临时文件。wav 直接返回原路径。"""
    if path.lower().endswith(".wav"):
        return path, None
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "24000", "-ac", "1", tmp.name], check=True)
    return tmp.name, tmp.name


def _syllables(text):
    """韩语音节数近似：去掉空白与常见标点后的字符数。

    ⚠️ 是近似，不是精确音节切分。但对照各方用的是**同一份文本**，
    所以近似误差对所有候选是同一个常数，比较仍然成立。
    """
    if not text:
        return None
    n = 0
    for ch in text:
        if ch.isspace() or ch in ".,;:!?~·…—-()[]{}\"'“”‘’《》<>/\\|":
            continue
        n += 1
    return n


def praet_abe(t, f, segs_sec, tail_sec=TAIL_SEC):
    """🔴 用 **Praat 口径** 一次算出 A / B / E —— 与 med / drop 同源。

    为什么不能再用 `prosody_report.measure` 的 A/B/E：
    那个走的是 librosa.pyin，而 med/drop 走 Praat（app/proso.tier_of）。
    2026-10-12 第一版把两套数放进同一张表 —— **跨口径数字不可比**，
    这是项目里反复踩过、明确写进 MEMORY 的坑。这里统一到 Praat。

    另一个好处：pyin 几分钟音频要 1~2 分钟，Praat 只要 1~2 秒，
    36 条案例的全库画像才能跑得动（否则必超时）。

    segs_sec 为 [(start,end),...]（秒）；为空时退化为整段一句。
    """
    f = np.asarray(f, dtype=np.float64)
    f = f[f > 0]
    if f.size < 8:
        return None, None, None
    semi_all = 12.0 * np.log2(f + 1e-12)
    # B：全局音高跨度
    B = float(np.percentile(semi_all, 95) - np.percentile(semi_all, 5))

    if not segs_sec:
        segs_sec = [(0.0, float(t[-1]) if len(t) else 0.0)]

    per, sl = [], []
    for s0, e0 in segs_sec[:80]:
        m = (t >= s0) & (t < e0)
        ff = f[m]
        if ff.size >= 5:
            per.append(12.0 * np.log2(np.percentile(ff, 85) /
                                      np.percentile(ff, 15) + 1e-12))
        a = max(s0, e0 - tail_sec)
        m2 = (t >= a) & (t < e0)
        if e0 - a >= 0.15:
            ft = f[m2]
            if ft.size >= 5:
                ft = ft[(ft > np.percentile(ft, 5)) &
                        (ft < np.percentile(ft, 95))]
                if ft.size >= 5:
                    semi = 12.0 * np.log2(ft / np.median(ft) + 1e-12)
                    sl.append(np.polyfit(np.arange(len(semi)), semi, 1)[0] *
                              len(semi))
    A = float(np.median(per)) if len(per) >= 2 else None
    E = float(np.std(sl)) if len(sl) >= 2 else None
    return A, B, E


def tail_drop(x, sr, is_male, segs, tail_sec=TAIL_SEC):
    """句尾定向降幅（半音）。负 = 降调，正 = 上扬。

    为什么单独量这一项：E（斜率标准差）只看"句尾是不是一个调"，
    看不出"整体降得够不够"。案例是新闻播报腔（K-ToBI HL%），
    句尾要有明确的落地感；我们 base 曾测出 −5.02 st vs 案例 −7.62 st。
    """
    t, f = proso.tier_of(x, sr, is_male=is_male)
    if t is None:
        return None
    segs_sec = [(s / float(sr), e / float(sr)) for s, e in segs[:80]]
    return _drop_from(t, f, segs_sec, tail_sec)


def cdf_of(x, sr, cuts):
    """C 能量动态 / D 语速CV / F 停顿熵 —— 只用 RMS，不碰 F0。

    ⚠️ 为什么不用 `prosody_report.measure`：它为了算 A/B/E 会跑一遍
    librosa.pyin，几分钟的音频要 1~2 分钟。全库 36 条案例跑下来必超时。
    而 C/D/F 与 F0 无关，这里自己算，口径与 prosody_report 完全一致
    （同样的 50ms/25ms 帧、同样的 p10~p90 / 3 秒窗 / 8 桶直方图）。
    """
    x = np.asarray(x, dtype=np.float64)
    W, H = int(0.05 * sr), int(0.025 * sr)
    n = (len(x) - W) // H
    if n < 20:
        return None, None, None
    e = np.sqrt(np.array([np.mean(x[i * H:i * H + W] ** 2)
                          for i in range(n)]) + 1e-12)
    edb = 20 * np.log10(e + 1e-12)
    voiced = edb[edb > np.percentile(edb, 25)]
    C = (float(np.percentile(voiced, 90) - np.percentile(voiced, 10))
         if len(voiced) > 20 else None)

    fps = 1 / (H / sr)
    d = np.clip(np.diff(e, prepend=e[0]), 0, None)
    k = max(int(0.02 * fps), 1)
    d = np.convolve(d, np.ones(k) / k, mode="same")
    ab = d > np.percentile(d, 70)
    Wn = int(3.0 * fps)
    rates, louds = [], []
    for i in range(0, len(ab) - Wn, Wn):
        rates.append(np.sum((ab[i + 1:i + Wn]) &
                            (~ab[i:i + Wn - 1])) / 3.0)
        louds.append(np.mean(e[i:i + Wn]))
    r, lo_ = np.array(rates), np.array(louds)
    if len(r) > 3:
        r = r[lo_ > np.percentile(lo_, 15)]
    D = float(r.std() / r.mean()) if len(r) > 3 and r.mean() > 0 else None

    runs = np.array([(j - i) / sr for i, j in cuts])
    runs = runs[runs >= 0.28]
    F = None
    if len(runs) >= 5:
        hist = np.histogram(runs, bins=8)[0].astype(float)
        prob = hist / hist.sum()
        prob = prob[prob > 0]
        F = float(-(prob * np.log2(prob)).sum() / np.log2(8))
    return C, D, F


def _drop_from(t, f, segs_sec, tail_sec=TAIL_SEC):
    """给定已算好的 PitchTier，算句尾定向降幅（半音）。

    抽出来是为了让 `measure()` 里 t/f 只取一次、四处复用 —— 全库 36 条案例
    的画像提取靠这个才不会超时（pyin 口径跑一遍要 1 小时以上）。
    """
    drops = []
    for s0, e0 in segs_sec[:80]:
        a = max(s0, e0 - tail_sec)
        if e0 - a < 0.15:
            continue
        m = (t >= a) & (t < e0)
        if m.sum() < 5:
            continue
        ff = np.asarray(f, dtype=np.float64)[m]
        ff = ff[(ff > np.percentile(ff, 10)) & (ff < np.percentile(ff, 90))]
        if len(ff) < 5:
            continue
        semi = 12.0 * np.log2(ff + 1e-12)
        drops.append(np.polyfit(np.arange(len(semi)), semi, 1)[0] * len(semi))
    return float(np.median(drops)) if len(drops) >= 2 else None


def measure(path, is_male, text=None, name=""):
    """全部走 **Praat 口径**，不混 pyin。

    C / D / F 与 F0 无关，仍复用 prosody_report（RMS 口径，同源）；
    A / B / E / med / drop 一律走 app/proso.tier_of 的 PitchTier。
    ⚠️ 只有 t/f 取一次、四处复用，才跑得动 36 条案例的全库画像。
    """
    x, sr = PR.load(path)
    segs, cuts, hop = PR.sent_windows(x, sr)
    c, d, f_ = cdf_of(x, sr, cuts)
    m = {"name": name or os.path.basename(path), "C": c, "D": d, "F": f_}
    segs_sec = [(s / float(sr), e / float(sr)) for s, e in segs]
    dur = len(x) / float(sr)

    t, f = proso.tier_of(x, sr, is_male=is_male)
    if t is None:
        m.update({"med": None, "drop": None, "A": None, "B": None, "E": None})
    else:
        v = np.asarray(f, dtype=np.float64)
        v = v[v > 0]
        m["med"] = float(np.median(v)) if v.size >= 8 else None
        m["drop"] = _drop_from(t, f, segs_sec)
        m["A"], m["B"], m["E"] = praet_abe(t, f, segs_sec)

    m["prate"] = (len(cuts) / dur * 60.0) if dur > 1.0 else None
    syl = _syllables(text) if text else None
    m["sps"] = (syl / dur) if syl else None
    m["dur"] = dur
    return m


KEYS = [("med", "音高中位", 0), ("A", "句内动态", 0), ("B", "音高跨度", 0),
        ("C", "能量动态", 0), ("D", "语速CV", 0), ("E", "句尾STD", 0),
        ("F", "停顿熵", 0), ("drop", "句尾降幅", 0), ("sps", "语速", 0),
        ("prate", "停顿/分", 0)]


def _f(v, w=7, d=2):
    return ("%*.2f" % (w, v)) if isinstance(v, (int, float)) else " " * (w - 1) + "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="JSON 候选清单")
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    spec = json.load(open(a.spec, encoding="utf-8"))
    rows = []
    for it in spec:
        p = it["path"]
        if not os.path.exists(p):
            print("跳过（不存在）:", p)
            continue
        w, tmp = _to_wav(p)
        tf = it.get("text_file")
        txt = it.get("text")
        if tf and os.path.exists(tf):
            txt = open(tf, encoding="utf-8").read()
        try:
            m = measure(w, it.get("is_male"), txt,
                        name=it.get("name") or os.path.basename(p))
        finally:
            if tmp:
                os.unlink(tmp)
        m["_ref"] = bool(it.get("ref"))
        rows.append(m)

    if not rows:
        print("没有可用样本")
        return 1
    ref = next((r for r in rows if r["_ref"]), rows[0])

    hdr = "%-24s" % "样本"
    for k, lbl, _ in KEYS:
        hdr += " %s" % lbl.rjust(7)
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = "%-24s" % r["name"][:24]
        for k, _, _ in KEYS:
            line += " %s" % _f(r.get(k))
        print(line + ("   ← 参照" if r is ref else ""))

    print("\n--- 相对参照的偏差（%+）---")
    print("%-24s" % "样本" + "".join(" %s" % lbl.rjust(7)
                                     for _, lbl, _ in KEYS))
    for r in rows:
        if r is ref:
            continue
        line = "%-24s" % r["name"][:24]
        for k, _, _ in KEYS:
            v, rv = r.get(k), ref.get(k)
            if isinstance(v, (int, float)) and isinstance(rv, (int, float)) and rv:
                line += " %+7.2f" % (v - rv)
            elif isinstance(v, (int, float)) and isinstance(rv, (int, float)):
                line += " %+7.2f" % (v - rv)
            else:
                line += " " * 7 + "-"
        print(line)

    if a.json_out:
        json.dump([dict([(k, r.get(k)) for k, _, _ in KEYS] +
                        [("name", r["name"])])
                   for r in rows],
                  open(a.json_out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print("\n已写", a.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
