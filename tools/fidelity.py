# -*- coding: utf-8 -*-
r"""还原度评分 v3（fidelity3）—— 量「平台成品」与「案例原声」的差距。

为什么必须放进仓库
------------------
2026-10-07 服务器重启把 /tmp 清空，上一版 `fidelity3.py` / `eval5r20.py` /
`submit5.py` 全部丢失，r20 的 93.52% 基线一度无法复现。评分口径是项目的
**验收标准**，不能只存在于临时目录，所以这里按文档反推重建并固化。

口径来源（反推自 docs/r18-五案例线上验收-2026-10-07.md 的逐项偏差表，
用 5 个案例的分数做闭合验证，11 项中 9 项可直接对上，残差落在 D 一项内）：

    还原度 = 100 − Σ 权重 × 偏差(%)

| 维度           | 权重 | 偏差定义                              |
|----------------|------|---------------------------------------|
| A 句内音高动态 | 0.10 | 相对偏差 \|o−c\|/c                    |
| B 全局音高跨度 | 0.08 | 相对偏差                              |
| E 句尾多样性   | 0.12 | 相对偏差                              |
| D_voiced 语速CV| 0.08 | 相对偏差（按有声时间归一）            |
| F 停顿熵       | 0.08 | 相对偏差                              |
| 静音占比       | 0.07 | 相对偏差                              |
| C 能量动态     | 0.08 | 相对偏差                              |
| 时长           | 0.10 | 相对偏差                              |
| LUFS           | 0.06 | \|ΔdB\| / 6 × 100（6 dB = 完全不像）  |
| 峰值 dBFS      | 0.06 | \|ΔdB\| / 6 × 100                     |
| 音色 LTAS      | 0.07 | \|ΔdB\| / 6 × 100（<11kHz 平均绝对）  |

权重合计 0.90（不是 1.0）—— 这是历史口径，**不要归一化**，否则分数对不上。
CER 不参与打分，只作为闸门（> 噪声底 0.2227 说明内容跑飞）。

🔴 三条硬约束
1. 必须在 **qwen 环境** 跑（librosa 1.0.0）。cosy 环境是 librosa 0.10.2，
   pyin 口径不同 → F0 类指标全部不可比（测量陷阱 #1）。
2. 量的是 **平台成品 mp3**（解码后等价），不能再做任何后处理。
   离线 finalize.py 里的 alimiter 会把 E 打掉 18%（陷阱 #16）。
3. **跨长度不可比**：逐句统计量只在同文本/同长度下可比（验收铁律）。
"""
import os
import re
import json
import subprocess
import sys

import numpy as np
import soundfile as sf
import librosa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.prosody_report import load, sent_windows, f0_of  # noqa: E402

# dB 类指标的「完全不像」跨度：差 6 dB 视为 100% 偏差
DB_FULL_SCALE = 6.0
# LTAS 只比 11kHz 以下（我们 24k 输出在 12k 截止，再往上是编码噪声）
LTAS_FMAX = 11000.0

WEIGHTS = {
    "A": 0.10, "B": 0.08, "E": 0.12, "D": 0.08, "F": 0.08,
    "sil": 0.07, "C": 0.08, "dur": 0.10,
    "lufs": 0.06, "peak": 0.06, "tone": 0.07,
}

# ---------------- 响度 / 峰值 ----------------

def loudness(path):
    """ffmpeg ebur128 实测：返回 (integrated LUFS, true peak dBFS)。

    与平台闭环用的是同一个 ffmpeg 口径，保证同源可比。
    """
    try:
        p = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "info", "-i", path,
             "-af", "ebur128=peak=true", "-f", "null", "-"],
            capture_output=True, text=True, timeout=900)
        out = p.stdout + p.stderr
    except Exception:
        return None, None
    tail = out[out.rfind("Summary:"):] if "Summary:" in out else out
    m = re.search(r"I:\s*(-?[\d.]+)\s*LUFS", tail)
    lufs = float(m.group(1)) if m else None
    peaks = re.findall(r"Peak:\s*(-?[\d.]+)\s*dBFS", tail)
    peak = max(float(x) for x in peaks) if peaks else None
    return lufs, peak


# ---------------- LTAS ----------------

def _mel_fb(sr, n_fft=2048, n_mels=128):
    fmin, fmax = 50.0, sr / 2.0
    return (librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                                fmin=fmin, fmax=fmax),
            librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax))


# EQ 的 1/3 倍频程中心频率（app/server.py superequalizer 的前 15 段）。
# 11840Hz 以上一律不算：我们 24k 输出在 12k 就截止，案例 44.1k mp3 在那里
# 的能量只是编码噪声。
EQ_BANDS = [65, 92, 131, 185, 262, 370, 523, 740, 1047, 1480, 2093,
            2960, 4186, 5920, 8372]


def ltas_db(x, sr=24000, n_fft=2048, hop=512, n_mels=128, pct=35):
    """长时平均谱（mel 域 dB），已按**总能量归一**。

    🔴 三个参数都是口径的一部分，改任何一个数值都会跳（实测 p50→p35 让
    标杆男音色从 1.70 dB 掉到 0.75 dB）：
      * `pct=35` 只取有声帧（帧能量高于 p35）。用 p50 会把大量低能量帧
        （含我们 −120 dB 的数字静音）算进平均，高频被静音拉偏 3 dB。
      * 归一到总能量（测量陷阱 #13）——不归一的话「整体轻」会被误读成
        「低频少」。整体电平由 LUFS 那一项单独处理。
      * 只比 60Hz~11kHz（11.2k 以上是 lame 的刀，不可比）。
    """
    if x.size < int(sr * 0.3):
        return None, None
    S = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop)) ** 2
    fb, mel_f = _mel_fb(sr, n_fft, n_mels)
    M = fb @ S
    if M.shape[1] < 3:
        return None, None
    e = M.mean(axis=0)
    sel = e > np.percentile(e, pct)
    if int(sel.sum()) < 3:
        sel = np.ones(M.shape[1], dtype=bool)
    p = M[:, sel].mean(axis=1)
    db = 10 * np.log10(p / (p.sum() + 1e-20) + 1e-12)
    return db, mel_f


def tone_dev(db_a, mel_f_a, db_b, mel_f_b):
    """两条 LTAS 在 1/3 倍频程（= EQ 的 15 段）上的平均绝对偏差（dB）。

    用 1/3 倍频程而不是逐个 mel 通道：mel 通道内的锯齿状起伏是内容差异
    （不同音素），不是音色差异，逐通道平均会把噪声当信号。
    """
    if db_a is None or db_b is None:
        return None
    devs = []
    for lo_, hi_ in zip(EQ_BANDS, EQ_BANDS[1:] + [LTAS_FMAX]):
        sa = (mel_f_a >= lo_) & (mel_f_a < hi_)
        sb = (mel_f_b >= lo_) & (mel_f_b < hi_)
        if sa.sum() == 0 or sb.sum() == 0:
            continue
        devs.append(float(np.mean(db_a[sa]) - np.mean(db_b[sb])))
    if not devs:
        return None
    return float(np.mean(np.abs(devs)))


# ---------------- 六维 + 静音 ----------------

def measure(path, name=""):
    """量一个音频文件，返回全部指标。"""
    x, sr = load(path, sr0=24000)
    segs, cuts, hop = sent_windows(x, sr)

    # A / B
    per, allf = [], []
    for s, e in segs[:80]:
        f = f0_of(x[s:e], sr)
        if f is None:
            continue
        allf.append(f)
        per.append(12 * np.log2(np.percentile(f, 85) / np.percentile(f, 15) + 1e-12))
    A = float(np.median(per)) if len(per) >= 2 else None
    B = None
    if allf:
        g = np.concatenate(allf)
        B = float(12 * np.log2(np.percentile(g, 95) / np.percentile(g, 5) + 1e-12))

    # C 能量动态 + 静音占比
    W, H = int(0.05 * sr), int(0.025 * sr)
    n = (len(x) - W) // H
    e = np.sqrt(np.array([np.mean(x[i * H:i * H + W] ** 2) for i in range(n)]) + 1e-12)
    edb = 20 * np.log10(e + 1e-12)
    thr = np.percentile(edb, 25)
    voiced = edb[edb > thr]
    C = float(np.percentile(voiced, 90) - np.percentile(voiced, 10)) if len(voiced) > 20 else None
    voiced_ratio = float((edb > thr).mean())

    # D_voiced：3 秒窗 onset 密度 **按窗内有声占比归一** 后的变异系数
    fps = 1 / (H / sr)
    d = np.clip(np.diff(e, prepend=e[0]), 0, None)
    k = max(int(0.02 * fps), 1)
    d = np.convolve(d, np.ones(k) / k, mode="same")
    ab = d > np.percentile(d, 70)
    Wn = int(3.0 * fps)
    rates, louds = [], []
    for i in range(0, len(ab) - Wn, Wn):
        rates.append(np.sum((ab[i + 1:i + Wn]) & (~ab[i:i + Wn - 1])) / 3.0)
        louds.append(np.mean(e[i:i + Wn]))
    r, lo_ = np.array(rates), np.array(louds)
    if len(r) > 3:
        r = r[lo_ > np.percentile(lo_, 15)]
    D = float(r.std() / r.mean()) if len(r) > 3 and r.mean() > 0 else None
    D_voiced = D / max(voiced_ratio, 1e-6) if D is not None else None

    # E 句尾多样性
    slopes = []
    for s, e2 in segs[:80]:
        a = max(s, e2 - int(0.35 * sr))
        if e2 - a < int(0.15 * sr):
            continue
        f = f0_of(x[a:e2], sr)
        if f is None:
            continue
        ff = f[(f > np.percentile(f, 5)) & (f < np.percentile(f, 95))]
        if len(ff) < 5:
            continue
        semi = 12 * np.log2(ff / np.median(ff) + 1e-12)
        slopes.append(np.polyfit(np.arange(len(semi)), semi, 1)[0] * (sr / 256))
    E = float(np.std(slopes)) if len(slopes) >= 2 else None

    # F 停顿熵（cuts 已是样本索引，见 prosody_report 里的 480 倍 bug 说明）
    runs = np.array([(j - i) / sr for i, j in cuts])
    runs = runs[runs >= 0.28]
    F = None
    if len(runs) >= 5:
        hist = np.histogram(runs, bins=8)[0].astype(float)
        prob = hist / hist.sum()
        prob = prob[prob > 0]
        F = float(-(prob * np.log2(prob)).sum() / np.log2(8))

    # 静音占比 = 检出的停顿总时长 / 总时长（不是帧级 VAD 掩码）。
    # 口径校验：案例 1a6c20 算出来 0.1875，与 docs/r18 §7 记录的 0.188 一致。
    # ⚠️ 别用帧级相对阈值（p25+6）去算 —— 那是相对阈值，两边都会被推到
    #    差不多 0.33~0.39，且对底噪敏感，跟历史数值对不上。
    sil_ratio = float(sum((j - i) for i, j in cuts) / max(len(x), 1))

    lufs, peak = loudness(path)
    ldb, mfreq = ltas_db(x, sr)

    return dict(name=name or os.path.basename(path),
                A=A, B=B, C=C, D=D_voiced, E=E, F=F,
                sil=sil_ratio, dur=float(len(x) / sr),
                lufs=lufs, peak=peak,
                voiced_ratio=voiced_ratio,
                ltas=(ldb.tolist() if ldb is not None else None),
                mel_f=(mfreq.tolist() if mfreq is not None else None),
                segs=len(segs), pauses=len(runs))


# ---------------- 评分 ----------------

def _rel(o, c):
    if o is None or c is None or abs(c) < 1e-9:
        return None
    return abs(o - c) / abs(c) * 100.0


def _dbdev(o, c):
    if o is None or c is None:
        return None
    return abs(o - c) / DB_FULL_SCALE * 100.0


def score(case, prod):
    """返回 (总分, 逐项偏差 dict)。case/prod 均为 measure() 的结果。"""
    devs = {
        "A": _rel(prod["A"], case["A"]),
        "B": _rel(prod["B"], case["B"]),
        "E": _rel(prod["E"], case["E"]),
        "D": _rel(prod["D"], case["D"]),
        "F": _rel(prod["F"], case["F"]),
        "sil": _rel(prod["sil"], case["sil"]),
        "C": _rel(prod["C"], case["C"]),
        "dur": _rel(prod["dur"], case["dur"]),
        "lufs": _dbdev(prod["lufs"], case["lufs"]),
        "peak": _dbdev(prod["peak"], case["peak"]),
    }
    ca = case.get("ltas")
    pa = prod.get("ltas")
    if ca and pa:
        devs["tone"] = _dbdev(
            tone_dev(np.array(pa), np.array(prod["mel_f"]),
                     np.array(ca), np.array(case["mel_f"])), 0.0)
    else:
        devs["tone"] = None

    loss = 0.0
    for k, w in WEIGHTS.items():
        v = devs.get(k)
        if v is None:
            continue
        loss += w * min(v, 100.0)
    return round(100.0 - loss, 2), devs


def _fmt(v, d=2):
    return ("%.*f" % (d, v)) if v is not None else "  -"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--prod", required=True)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    cm = measure(a.case, "CASE")
    pm = measure(a.prod, "PROD")
    s, devs = score(cm, pm)
    print(f"{'维度':<14}{'权重':>6}{'案例':>10}{'成品':>10}{'偏差%':>9}{'失分':>8}")
    for k in ["A", "B", "E", "D", "F", "sil", "C", "dur", "lufs", "peak", "tone"]:
        print(f"{k:<14}{WEIGHTS[k]:>6.2f}{_fmt(cm.get(k)):>10}{_fmt(pm.get(k)):>10}"
              f"{_fmt(devs.get(k)):>9}{_fmt(WEIGHTS[k] * (devs.get(k) or 0), 3):>8}")
    print(f"\n还原度 = {s}%   时长 {cm['dur']:.1f} / {pm['dur']:.1f} s")
    if a.json:
        json.dump({"case": cm, "prod": pm, "devs": devs, "score": s},
                  open(a.json, "w"), ensure_ascii=False)
