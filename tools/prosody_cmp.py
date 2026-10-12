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


def tail_drop(x, sr, is_male, segs, tail_sec=TAIL_SEC):
    """句尾定向降幅（半音）。负 = 降调，正 = 上扬。

    为什么单独量这一项：E（斜率标准差）只看"句尾是不是一个调"，
    看不出"整体降得够不够"。案例是新闻播报腔（K-ToBI HL%），
    句尾要有明确的落地感；我们 base 曾测出 −5.02 st vs 案例 −7.62 st。
    """
    t, f = proso.tier_of(x, sr, is_male=is_male)
    if t is None:
        return None
    drops = []
    for s, e in segs[:80]:
        s0, e0 = s / float(sr), e / float(sr)
        a = max(s0, e0 - tail_sec)
        if e0 - a < 0.15:
            continue
        m = (t >= a) & (t < e0)
        if m.sum() < 5:
            continue
        ff = f[m]
        ff = ff[(ff > np.percentile(ff, 10)) & (ff < np.percentile(ff, 90))]
        if len(ff) < 5:
            continue
        semi = 12.0 * np.log2(ff + 1e-12)
        drops.append(np.polyfit(np.arange(len(semi)), semi, 1)[0] * len(semi))
    return float(np.median(drops)) if len(drops) >= 2 else None


def measure(path, is_male, text=None, name=""):
    x, sr = PR.load(path)
    segs, cuts, hop = PR.sent_windows(x, sr)
    m = PR.measure(path, name=name)
    dur = len(x) / float(sr)
    m["med"] = proso.med_of(x, sr, is_male=is_male)
    m["drop"] = tail_drop(x, sr, is_male, segs)
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
