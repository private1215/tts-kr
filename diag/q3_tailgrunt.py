# -*- coding: utf-8 -*-
"""q3_tailgrunt.py —— Q3：句末「嗯」到底是什么、能不能清、EL 为什么舒服。

三个文件同口径（tailgrunt53e）：
  我们成品 202610190902.mp3 / EL 标杆 merged.wav / 模版 EL_female.wav

做四件事：
1. 宽/严两口径计数（严=该段之后还有 >0.05s 静音，即真是「句末」；
   宽=只要是紧贴浊音段的小段就算——后者更接近员工说的「吐字后尾音」）；
2. 每个候选段的谱特征：时长 / 峰值 / 低频占比 / 谱平坦度 / 自相关谐波度 /
   H1-H2（判定「声带振动的自然释放」还是「噪声/口水」）；
3. 按时长档 × 峰值档做二维分布对比；
4. 看这些段出现的位置：句末（前段以标点结尾）还是短语中间。
"""
import json
import re
import subprocess

import numpy as np
import soundfile as sf

FILES = [
    ("我们成品 202610190902", "/mnt/tts/diag/v0902/202610190902-1.mp3.16k.wav", False),
    ("EL 标杆 merged", "/mnt/tts/diag/v0902/hm_merged.wav.wav", False),
    ("模版 EL_female（24k原生）", "/mnt/tts/data/voices/EL_female.wav", True),
]


def load(path, native):
    if native:
        x, sr = sf.read(path, dtype="float32")
    else:
        w = "/tmp/q3_%s.wav" % re.sub(r"\W", "_", path)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                        "-ar", "16000", "-ac", "1", w], check=True)
        x, sr = sf.read(w, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def segs_of(x, sr):
    """-40dB 分段。"""
    hop = int(0.005 * sr)
    n = len(x) // hop
    rms = np.array([np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2) + 1e-12)
                    for i in range(n)])
    db = 20 * np.log10(rms + 1e-12)
    voiced = db > -40
    segs, i = [], 0
    while i < n:
        if voiced[i]:
            j = i
            while j < n and voiced[j]:
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1
    return segs, hop


def feats(s, sr):
    """小段的谱/时序特征。"""
    if len(s) < 16:
        return None
    w = s * np.hanning(len(s))
    S = np.abs(np.fft.rfft(w)) ** 2
    fr = np.fft.rfftfreq(len(s), 1.0 / sr)
    tot = S.sum() + 1e-12
    lowf = float(S[fr < 300].sum() / tot)
    # 谱平坦度（几何/算术）：→1 是噪声，小的是谐波
    Spos = S[S > 0]
    flat = float(np.exp(np.mean(np.log(Spos + 1e-12))) / (np.mean(Spos) + 1e-12)) if len(Spos) else 1.0
    # 自相关谐波度：归一化自相关的第一个非零峰
    s2 = s - s.mean()
    ac = np.correlate(s2, s2, "full")[len(s2) - 1:]
    if ac[0] <= 0:
        harm = 0.0
    else:
        acn = ac / ac[0]
        lo, hi = int(0.004 * sr), int(0.016 * sr)   # 62~250Hz 对应 250~62 Hz 基频范围
        harm = float(np.max(acn[lo:hi])) if hi < len(acn) else 0.0
    zcr = float(np.mean(np.abs(np.diff(np.sign(s))) > 0))
    return {"lowf": lowf, "flat": flat, "harm": harm, "zcr": zcr}


def collect(x, sr, strict):
    segs, hop = segs_of(x, sr)
    out = []
    for k in range(1, len(segs)):
        pi, pj = segs[k - 1]
        ci, cj = segs[k]
        gap = (ci - pj) * 0.005
        if gap >= 0.03:
            continue
        ln = (cj - ci) * 0.005
        if not (0.005 <= ln <= 0.12):
            continue
        if (pj - pi) * 0.005 <= 0.10:
            continue
        if strict and (k + 1 >= len(segs) or (segs[k + 1][0] - cj) * 0.005 <= 0.05):
            continue
        s = x[ci * hop:cj * hop]
        if len(s) < 16:
            continue
        pk = float(np.max(np.abs(s)))
        pkdb = 20 * np.log10(pk + 1e-12)
        if not (-45 <= pkdb <= -20):
            continue
        f = feats(s, sr)
        if f is None:
            continue
        if f["zcr"] < 0.15 and f["lowf"] > 0.45:
            out.append({"t": ci * 0.005, "dur": ln, "pkdb": pkdb,
                        "pre_dur": (pj - pi) * 0.005, **f})
    return out


def dist_table(items, key1, key2, bins1, bins2, labels1, labels2):
    print("  %-22s" % "", end="")
    for l in labels2:
        print("%-12s" % l, end="")
    print()
    for i, l1 in enumerate(labels1):
        print("  %-22s" % l1, end="")
        for j, _ in enumerate(labels2):
            c = sum(1 for it in items
                    if bins1[i] <= it[key1] < bins1[i + 1] and bins2[j] <= it[key2] < bins2[j + 1])
            print("%-12s" % (c if c else "·"), end="")
        print()


def main():
    data = {}
    for lab, p, nat in FILES:
        x, sr = load(p, nat)
        dur = len(x) / sr
        strict = collect(x, sr, True)
        wide = collect(x, sr, False)
        data[lab] = (dur, strict, wide)
        print("=" * 78)
        print("%s（%.1f s）" % (lab, dur))
        print("  严口径（真句末，后接>0.05s 静音）：%3d 个 = %5.2f/分"
              % (len(strict), len(strict) / (dur / 60)))
        print("  宽口径（紧贴浊音段的小段都算）：%3d 个 = %5.2f/分"
              % (len(wide), len(wide) / (dur / 60)))
        for tag, items in (("严", strict), ("宽", wide)):
            if not items:
                continue
            d = np.array([it["dur"] * 1000 for it in items])
            pk = np.array([it["pkdb"] for it in items])
            hm = np.array([it["harm"] for it in items])
            lf = np.array([it["lowf"] for it in items])
            print("    [%s] 时长 中位%5.1fms p90%6.1fms max%6.1fms | 峰值 中位%6.1fdB min%6.1fdB"
                  % (tag, np.median(d), np.percentile(d, 90), d.max(),
                     np.median(pk), pk.min()))
            print("    [%s] 谐波度 中位%.3f | 低频占比 中位%.2f | >50ms 的段 %d 个"
                  % (tag, np.median(hm), np.median(lf), int((d > 50).sum())))

    # 二维分布：宽口径（最接近员工描述）
    print("\n" + "=" * 78)
    print("宽口径二维分布：行=时长档，列=峰值档（计数）")
    bins_d = [0, 10, 30, 50, 80, 130]
    lab_d = ["5-10ms", "10-30ms", "30-50ms", "50-80ms", "80-120ms"]
    bins_p = [-50, -35, -28, -22, -15]
    lab_p = ["<-35dB", "-35~-28", "-28~-22", ">-22dB"]
    for lab, (dur, strict, wide) in data.items():
        print("\n--- %s ---" % lab)
        dist_table(wide, "dur", "pkdb", bins_d, bins_p, lab_d, lab_p)

    # 谐波度 vs 时长的关系（判定浊音释放 vs 噪声）
    print("\n" + "=" * 78)
    print("谐波度（自相关峰，>0.5 视为有明显谐波=浊音）按时长档")
    for lab, (dur, strict, wide) in data.items():
        print("\n--- %s（宽口径 %d 个）---" % (lab, len(wide)))
        for i, l1 in enumerate(lab_d):
            grp = [it for it in wide if bins_d[i] <= it["dur"] * 1000 < bins_d[i + 1]]
            if not grp:
                continue
            hm = np.array([it["harm"] for it in grp])
            lf = np.array([it["lowf"] for it in grp])
            print("  %-10s n=%3d  谐波度中位%.3f（>0.5 占%.0f%%）  低频占比中位%.2f"
                  % (l1, len(grp), np.median(hm), 100 * float((hm > 0.5).mean()), np.median(lf)))


if __name__ == "__main__":
    main()
