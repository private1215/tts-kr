# -*- coding: utf-8 -*-
"""通用 σ 探针：男女共用，基线从 JSON 读，不再把数字抄进代码。

为什么要有它：
    eq_search6.py 的 BASE 是**手抄**的男声清洗值，女声根本没有对应物。
    抄进去的数字会腐烂（男声已经发生过一次：原始基线把脏数据算进去，
    代码里却是清洗值，两边对不上没人发现）。现在统一读 el_base_clean_*.json，
    el_baseline_clean.py 重算一次就能刷新全部脚本。

🔴🔴 2026-10-15 三处口径修正（都是实测发现的「尺子坏了」）
------------------------------------------------------------------
1. **60 秒硬截断**（原第 74 行 `y16[:60*16000]`）
   实证：同一份成品 `跨文案对照-男8197-2.mp3` 只改切片起点，
   Σz² 在 **211 ~ 877** 之间波动（4 倍），最差维度还从 150~300 变成 4k~8k。
   → 历史所有 Σz² 都是「前 60s」口径，不能当全篇水平。
   → 现在默认 **--full**（全篇），`--head60` 可复现旧口径（仅供对照历史数字）。

2. **σ 被离群案例撑大**（实测男 150~300 撑 9.6x、女 60~150 撑 26.8x）
   `el_baseline_clean.py` 只剔了 2 条显式脏数据，**没处理剩余离群值把 SD 撑大**。
   → 「|z| < 3 达标」是假达标：报 −3.08σ 的 150~300 在稳健尺度下应是 −26.7σ。
   → 现在**同时输出朴素 σ 与稳健 σ（MAD×1.4826）两套 z**，默认按稳健判。

3. **目标不该是 0**（实测 12 组 EL 案例切片：Σz² 17 ~ 240，中位 59.6）
   EL 标杆自己就拿 239.8。→ 现在输出 `--ref` 对照 EL 自检分布，
   判定目标是「**落进 EL 分布**」，不是趋近 0。

用法：
    python probe_g.py female /path/a.wav /path/b.wav       # 默认全篇+稳健σ
    python probe_g.py male m_v10.wav --head60               # 复现历史口径
    python probe_g.py male a.wav --segments                # 长篇沿时间轴稳定性
    python probe_g.py male a.wav --ref                     # 对照 EL 自检分布（合格线）
    python probe_g.py male a.wav --sigma naive             # 用朴素 σ

🔴🔴🔴 2026-10-15 用户铁律（优先级高于本脚本任何默认行为）
------------------------------------------------------------------
> **「如果只是几个字几十秒，不是长文案的生成语音，确实没什么参考意义，
>   短文案语音肯定都漂亮，长文案语音稳定才是实力。」**

**因此：本脚本的所有口径判断都必须建立在「长文案（≥3 分钟）」上。**
- 默认 `--full` 测全篇，这是唯一的主口径。
- ⚠️ **短切片（20~40s）的 Σz² 只能当"这一段怎么样"的定位信息，
  绝不能用来判断"这条链路行不行"** —— 短文案本来就漂亮，
  用它验收等于没验收，还会被生成波动骗（历史上多次翻车）。
- 长篇的正确用法：**测沿时间轴的稳定性**（`--segments`）——
  EL 标杆的 150~300 沿时间轴极差只有 16%，我们曾达 47%，
  **这个「一致性」差异才是长文案的真差距**，不是任何单点的绝对值。
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import librosa
import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
BANDS = [(60, 150), (150, 300), (300, 500), (500, 1000),
         (1000, 2000), (2000, 4000), (4000, 8000)]
ORDER = ["b60_150", "b150_300", "b300_500", "b500_1000",
         "b1000_2000", "b2000_4000", "b4000_8000", "flat", "voiced"]
NAME = {"b60_150": "60~150", "b150_300": "150~300", "b300_500": "300~500",
        "b500_1000": "500~1k", "b1000_2000": "1k~2k",
        "b2000_4000": "2k~4k", "b4000_8000": "4k~8k",
        "flat": "平坦度", "voiced": "发声占比"}

DIAG = "/mnt/tts/diag"
# 🔴 用户铁律 2026-10-15：短文案（几十秒）验收没有参考意义，
#    长文案（≥3 分钟）的「稳定」才是实力。低于此值只打警告，不下结论。
LONG_MIN = 180


def _dur(path):
    try:
        return float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path], capture_output=True,
            text=True, timeout=20).stdout.strip() or 0)
    except Exception:
        return 0


def base_path(gender, d=DIAG + "/x8197"):
    return os.path.join(d, "el_base_clean_%s.json" % gender)


def ref_path(gender, d=DIAG + "/selfcheck"):
    """EL 自检分布：标杆自己切片的 Σz²，用来定「合格线」。"""
    return os.path.join(d, "el_selfcheck_%s.json" % gender)


def load_base(gender):
    p = base_path(gender)
    if not os.path.exists(p):
        sys.exit("缺基线 %s，先跑 el_baseline_clean.py" % p)
    return json.load(open(p, encoding="utf-8"))


def load_ref(gender):
    p = ref_path(gender)
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def to24(path):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "24000", "-ac", "1", t], check=True)
    y, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    return y


def to16(path):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", "16000", "-ac", "1", t], check=True)
    y, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    return y


def _flat(y16):
    return float(librosa.feature.spectral_flatness(
        y=y16, hop_length=256)[0].mean())


def _voiced(y16):
    hop = int(0.01 * 16000)
    n = len(y16) // hop
    env = np.sqrt(np.array([(y16[i * hop:(i + 1) * hop] ** 2).mean()
                            for i in range(n)]))
    thr = np.percentile(env, 30) * 2.2
    return float((env > thr).mean())


def _measure(y24, y16):
    out = {}
    S = np.abs(librosa.stft(y24, n_fft=2048, hop_length=512)) ** 2
    f = librosa.fft_frequencies(sr=24000, n_fft=2048)
    tot = S.sum() + 1e-12
    for lo, hi in BANDS:
        out["b%d_%d" % (lo, hi)] = float(S[(f >= lo) & (f < hi)].sum() / tot)
    out["flat"] = _flat(y16)
    out["voiced"] = _voiced(y16)
    return out


def measure(path, mode="full"):
    """mode: 'full' 全篇（主口径）| 'head60' 复现历史口径 | 'segments' 逐窗

    🔴 用户铁律：判定必须基于长文案全篇。短切片只作定位。
    """
    y24 = to24(path)
    y16 = to16(path)
    if mode == "head60":
        return _measure(y24, y16[:60 * 16000])
    if mode == "full":
        return _measure(y24, y16)
    # segments：30s 窗逐段测（跳过首尾各 5%）
    # 对长文案重点看「沿时间轴的一致性」，不是任何单窗的绝对值
    n16 = len(y16)
    seglen = 30 * 16000
    pad = int(n16 * 0.05)
    segs = []
    s = pad
    while s + seglen < n16 - pad:
        segs.append(("%.0fs" % (s / 16000.0),
                     _measure(y24[int(s * 24000 / 16000):
                                  int((s + seglen) * 24000 / 16000)],
                              y16[s:s + seglen])))
        s += seglen
    return segs


def zscored(m, base, keys, sigma="robust"):
    zs = {}
    for k in keys:
        b = base[k]
        sd = b.get("sd_robust") if sigma == "robust" else None
        if not sd or sd <= 1e-12:
            sd = b["sd"]
        zs[k] = (m[k] - b["mean"]) / sd if sd > 1e-12 else 0.0
    return zs


def _print_one(tag, m, base, keys, zs, sigma, ref=None):
    j = sum(v ** 2 for v in zs.values())
    worst = max(zs, key=lambda k: abs(zs[k]))
    verdict = ""
    if ref:
        lo, hi = ref["p25"], ref["p75"]
        if j < ref["min"] or j > ref["max"]:
            verdict = "  <<超出 EL 分布(%.0f~%.0f)" % (ref["min"], ref["max"])
        elif lo <= j <= hi:
            verdict = "  <<落进 EL 中间50%%(%.0f~%.0f) ✅" % (lo, hi)
        else:
            verdict = "  <<落进 EL 分布边缘"
    print("-" * 74)
    print("%-22s Σz²=%7.1f   最差 %s %+.2fσ(%s)%s"
          % (tag, j, NAME.get(worst, worst), zs[worst], sigma, verdict))
    for k in keys:
        z = zs[k]
        flag = "[X]" if abs(z) > 2 else ("[!]" if abs(z) > 1.5 else "[ok]")
        print("  %-4s %-9s 我们 %9.4f  EL %9.4f  z=%+7.2f"
              % (flag, NAME.get(k, k), m[k], base[k]["mean"], z))


def _stability(segs, base, keys, sigma):
    """长文案真正的差距：沿时间轴的**一致性**，不是任何单点的绝对值。

    短文案本来就漂亮，长文案稳不稳才是实力（用户铁律 2026-10-15）。
    """
    print("-" * 74)
    print("  【长文案稳定性】跨 %d 个 30s 窗（这才是长篇的真差距）"
          % len(segs))
    print("  %-9s %10s %10s %9s" % ("维度", "中位", "极差%", "CV%"))
    rows = []
    for k in keys:
        v = np.array([m[k] for _, m in segs], float)
        med = np.median(v)
        if abs(med) < 1e-12:
            continue
        rows.append(((v.max() - v.min()) / abs(med) * 100,
                     NAME.get(k, k), med,
                     v.std() / abs(med) * 100))
    for _rng, nm, med, cv in sorted(rows, reverse=True):
        print("  %-9s %10.4f %9.1f%% %8.1f%%" % (nm, med, _rng, cv))
    js = [sum(v * v for v in zscored(m, base, keys, sigma).values())
          for _, m in segs]
    if js:
        print("  跨窗 Σz²：min=%.0f 中位=%.0f max=%.0f  波动=%.0f%%"
              % (min(js), np.median(js), max(js),
                 (max(js) - min(js)) / max(np.median(js), 1e-9) * 100))


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("gender", choices=["male", "female"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--head60", action="store_true",
                    help="复现历史口径：只测前 60 秒（默认全篇）")
    ap.add_argument("--segments", action="store_true",
                    help="逐 30s 窗输出，看沿时间轴的稳定性")
    ap.add_argument("--sigma", choices=["robust", "naive"], default="robust",
                    help="σ 尺度：稳健 MAD×1.4826（默认）/ 朴素")
    ap.add_argument("--ref", action="store_true",
                    help="对照 EL 自检分布（标杆自己考多少分）")
    a = ap.parse_args()

    mode = "head60" if a.head60 else ("segments" if a.segments else "full")
    base = load_base(a.gender)
    keys = [k for k in ORDER if k in base]
    ref = load_ref(a.gender) if a.ref else None

    sig = "朴素"
    if a.sigma == "robust":
        sds = [base[k].get("sd_robust") or base[k]["sd"] for k in keys]
        nsds = [base[k]["sd"] for k in keys]
        if any(s / n > 1.5 for s, n in zip(sds, nsds) if n > 1e-12):
            sig = "稳健"
    print("基线：%s  n=%d  σ尺度=%s  口径=%s"
          % (base_path(a.gender), max(base[k]["n"] for k in keys), sig, mode))

    if ref:
        print("EL 自检（标杆自己，n=%d）：Σz² min=%.1f p25=%.1f 中位=%.1f "
              "p75=%.1f max=%.1f"
              % (ref["n"], ref["min"], ref["p25"], ref["median"],
                 ref["p75"], ref["max"]))
        print("  ➜ 判定目标 = 落进这个区间，不是趋近 0\n")

    for p in a.paths:
        p = p if os.path.isabs(p) else os.path.join(HERE, p)
        if not os.path.exists(p):
            print("缺文件 %s" % p)
            continue
        r = measure(p, mode)
        if mode == "segments":
            print("=" * 74)
            print("%s  逐窗稳定性" % os.path.basename(p))
            for tag, m in r:
                _print_one(tag, m, base, keys,
                           zscored(m, base, keys, a.sigma), sig, ref)
            _stability(r, base, keys, a.sigma)
        else:
            # 🔴 用户铁律：短文案验收没有参考意义，长篇才有
            dur = _dur(p)
            if dur < LONG_MIN and dur > 0:
                print("  ⚠️ 警告：时长仅 %.0fs（<%ds）。短文案本来就漂亮，"
                      "用它验收等于没验收。" % (dur, LONG_MIN))
            _print_one(os.path.basename(p)[:22], r, base, keys,
                       zscored(r, base, keys, a.sigma), sig, ref)


if __name__ == "__main__":
    main()
