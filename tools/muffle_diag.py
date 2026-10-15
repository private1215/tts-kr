# -*- coding: utf-8 -*-
"""🔴 「闷」的成因排查 —— 为什么方案换来换去还是这个状态。

## 背景（2026-10-15）
第 21 轮员工反馈里最硬的一条：
    第 6 题「最影响不像真人的是哪点」→ **A = 还是闷、像感冒鼻塞**
而 150~300Hz 频段已经从 **+3.96σ（过多）压到 −3.08σ（偏少）**，
员工**照样说闷**。第 2/3 题他还选了「两段差不多 / 说不清」。

也就是说：**我们把机器指标里唯一被叫做「闷」的那个频段，跨过了目标值，
听感一点没动。** 说明「闷」根本不在我们测量的坐标轴上。

## 本脚本要回答的问题
我们现有的验收体系（tools/probe_g.py）只有 **9 个全局标量**：
    7 个频带能量 + 平坦度(flat) + 发声占比(voiced)
全库 36 条 EL 案例也只提取了这 12 个标量（el_feats.json）。

而「闷 / 感冒鼻塞」在语音感知上对应的是一组**完全不同的物理量**——
它们全部是**时域逐帧**或**谐波结构**的量，一个都没进过我们的坐标系：

    HNR（谐噪比）   气声/沙哑 ← 听起来「像感冒」的第一位嫌疑
    F1/F2 轨迹      声道形状 ← 鼻音（感冒鼻塞）= F1 低、F2 高
    LTAS 斜率       整体明暗（不是单个频带）
    频谱质心        明亮度
    谱包络方差      音色扁平度 ← 听起来「像念稿/发假」

## 方法
对**同一篇 EL 案例**和**我们现役男声成品**做逐帧对齐分析，
输出「9 个标量之外」的维度差异。如果这些维度差得比 9 标量更远，
就证明**不是拆解不够细，而是拆解维度选错了**。

## 输出
/mnt/tts/diag/muffle/ 下的 JSON + 控制台摘要
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

SR = 16000
OUT_DIR = "/mnt/tts/diag/muffle"


def to16(path):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                    "-ar", str(SR), "-ac", "1", t], check=True)
    import soundfile as sf
    y, _ = sf.read(t, dtype="float32")
    os.unlink(t)
    return y


def frame_hnr(y, sr=SR, hop=256, win=1024):
    """逐帧 HNR：用自相关峰值幅度 / 残差能量 估谐噪比（dB）。

    这是语音质量分析的标准做法（Boersma 1993 的简化版）：
    自相关峰高 ≈ 周期成分能量，残差 ≈ 非周期成分。
    """
    n = len(y) // hop
    hnr = np.zeros(n)
    w = np.hanning(win)
    for i in range(n):
        fr = y[i * hop:i * hop + win]
        if len(fr) < win:
            break
        fr = fr - fr.mean()
        e = fr * w
        ac = np.correlate(fr, fr, mode="full")[len(fr) - 1:]
        if len(ac) < 10 or ac[0] <= 0:
            continue
        # 只在合理基频范围内找峰（男 70~350Hz，女 150~450Hz）
        lo, hi = int(sr / 350), int(sr / 70)
        hi = min(hi, len(ac) - 1)
        if hi <= lo:
            continue
        pk = lo + int(np.argmax(ac[lo:hi]))
        r = ac[pk] / ac[0]
        if r <= 0 or r >= 1.0:
            hnr[i] = 0.0
        else:
            hnr[i] = 10 * np.log10(r / (1.0 - r))
    return hnr


def fmt_bands(y, sr=SR, nfft=2048, hop=512):
    """LTAS 七频带 + 平坦度 + 质心（现有体系之外/之内都有，便于对照）"""
    win = np.hanning(nfft)
    frames = []
    for i in range(0, len(y) - nfft, hop):
        fr = y[i:i + nfft] * win
        S = np.abs(np.fft.rfft(fr)) ** 2 + 1e-20
        frames.append(S)
    if not frames:
        return None
    P = np.mean(frames, axis=0)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    bands = {}
    for lo, hi in [(60, 150), (150, 300), (300, 500), (500, 1000),
                   (1000, 2000), (2000, 4000), (4000, 8000)]:
        m = (freqs >= lo) & (freqs < hi)
        bands["b%d_%d" % (lo, hi)] = float(np.mean(P[m])) if m.any() else 0.0
    Ps = P / P.sum()
    nz = Ps[Ps > 0]
    flat = float(np.exp(np.log(nz).mean()) / nz.mean())
    cent = float(np.sum(freqs * Ps) / Ps.sum())
    # LTAS 斜率：1000Hz 以上的平均每倍频程下降量
    def bandmean(f):
        m = (freqs >= f) & (freqs < f * 2)
        return float(np.mean(P[m])) if m.any() else np.nan
    slope = np.log2(bandmean(4000) / bandmean(4000) + 1e-20)
    return {"bands": bands, "flat": flat, "centroid": cent,
            "rms_db": float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12))}


def formant_track(y, sr=SR):
    """用 LPC（Levinson-Durbin）估每 5ms 一帧的 F1/F2
    —— 鼻音（感冒鼻塞）= F1 偏低 + F2 偏高。

    只做粗估，够用来对比「我们 vs EL」的相对差异。
    """
    win, hop, order = 512, 80, 20   # 32ms / 5ms / LPC 阶 20
    out = []
    for i in range(0, len(y) - win, hop):
        fr = y[i:i + win]
        if np.sqrt(np.mean(fr ** 2)) < 0.005:      # 静音跳过
            out.append((np.nan, np.nan))
            continue
        fr = fr * np.hamming(win)
        try:
            r = np.correlate(fr, fr, "full")[len(fr) - 1:]
            r_ = np.array([r[k] if k < len(r) else 0.0
                           for k in range(order + 1)])
            if r_[0] <= 0:
                out.append((np.nan, np.nan))
                continue
            a = np.zeros(order)
            err = r_[0]
            for m in range(order):
                acc = r_[m + 1] - np.dot(a[:m], r_[m::-1])
                if err <= 1e-20:
                    break
                k = acc / err
                k = max(-0.99, min(0.99, k))
                a[m] = k
                if m > 0:
                    a[:m] = a[:m] - k * a[m - 1::-1]
                err *= (1 - k * k)
            roots = np.roots(np.concatenate([[1.0], -a]))
            roots = roots[np.imag(roots) > 0.01]
            f = np.sort(np.angle(roots) * sr / (2 * np.pi))
            f = f[np.logical_and(f > 90, f < 5000)]
            out.append((f[0] if len(f) else np.nan,
                        f[1] if len(f) > 1 else np.nan))
        except Exception:
            out.append((np.nan, np.nan))
    if not out:
        return None
    arr = np.array(out)
    f1 = arr[:, 0][np.isfinite(arr[:, 0])]
    f2 = arr[:, 1][np.isfinite(arr[:, 1])]
    if len(f1) == 0 or len(f2) == 0:
        return {"f1_med": None, "f2_med": None, "f1_mean": None,
                "f2_mean": None, "f1_iqr": None, "f2_iqr": None,
                "f1_n": int(len(f1)), "f2_n": int(len(f2))}
    return {
        "f1_med": float(np.median(f1)),
        "f2_med": float(np.median(f2)),
        "f1_mean": float(np.mean(f1)),
        "f2_mean": float(np.mean(f2)),
        "f1_iqr": float(np.percentile(f1, 75) - np.percentile(f1, 25)),
        "f2_iqr": float(np.percentile(f2, 75) - np.percentile(f2, 25)),
        "f1_n": int(len(f1)), "f2_n": int(len(f2)),
    }


def analyze(path, label):
    y = to16(path)
    r = {"label": label, "path": path, "dur": len(y) / SR}
    b = fmt_bands(y)
    if b:
        r.update(b)
    h = frame_hnr(y)
    hv = h[h > 0]
    r["hnr_med"] = float(np.median(hv)) if len(hv) else None
    r["hnr_p25"] = float(np.percentile(hv, 25)) if len(hv) else None
    r["hnr_frames"] = int(len(hv))
    f = formant_track(y)
    if f:
        r.update(f)
    else:
        r["formant"] = "unavailable"
    return r


def main():
    pairs = json.load(open(sys.argv[1], encoding="utf-8"))
    os.makedirs(OUT_DIR, exist_ok=True)
    res = []
    for p in pairs:
        try:
            res.append(analyze(p["path"], p["label"]))
            print("  ok:", p["label"], flush=True)
        except Exception as e:
            print("  FAIL:", p["label"], repr(e), flush=True)
    out = os.path.join(OUT_DIR, "muffle_diag.json")
    json.dump(res, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n=== 写入", out)
    # 控制台摘要
    keys = ["hnr_med", "hnr_p25", "centroid", "flat", "f1_med", "f2_med",
            "f2_iqr", "rms_db"]
    print("%-14s" % "label", " ".join("%12s" % k for k in keys))
    for r in res:
        print("%-14s" % r["label"][:14],
              " ".join("%12s" % (("%.4g" % r[k]) if isinstance(r.get(k), float)
                                 else str(r.get(k, "-"))) for k in keys))


if __name__ == "__main__":
    main()