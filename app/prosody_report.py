# -*- coding: utf-8 -*-
"""多维「情感丰富度」体检 —— 用来对齐 ElevenLabs 的听感。

用法：
    python app/prosody_report.py a.wav b.wav ...

为什么需要六个维度而不是一个：
    本项目踩过两次坑——只看「句内音高动态」会得出与实际听感相反的结论
    （指标上 Qwen 8.07 比 CosyVoice3 7.62 更接近 ElevenLabs 8.20，
     但员工主观说 Qwen 平淡）。单一聚合指标抓不到「人味」，必须多维度对照。

六个维度（同一脚本口径内才可比，**跨脚本/跨音频长度不可比**）：
    A 句内动态(semi)   每句 F0 p15~p85 宽度的中位 —— 「有没有声调」
    B 全局音高跨度     整段 F0 p5~p95 —— 「用了多少音域」
    C 能量动态(dB)     RMS p10~p90 —— 「有没有轻重音」
    D 语速变化 CV      3 秒窗音节密度变异系数 —— 「语速是不是一路到底」
    E 句尾多样性       每句末段斜率的标准差 —— 「尾声是不是一个调」
    F 停顿熵           停顿时长分布熵(0~1) —— 「停顿是不是只有一个长度」
"""
import sys
import numpy as np
import soundfile as sf
import librosa


def load(p, sr0=24000):
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != sr0:
        x = librosa.resample(x.astype(np.float64), orig_sr=sr,
                             target_sr=sr0).astype(np.float32)
        sr = sr0
    return x, sr


def sent_windows(x, sr, min_sil=0.25, min_len=0.6, max_len=30.0):
    hop, win = int(0.02 * sr), int(0.05 * sr)
    n = (len(x) - win) // hop
    idx = np.arange(n) * hop
    rms = np.sqrt(np.array([np.mean(x[i:i + win] ** 2) for i in idx]) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    sil = db < (np.percentile(db, 25) + 6.0)
    cuts, i = [], 0
    while i < n:
        if sil[i]:
            j = i
            while j < n and sil[j]:
                j += 1
            if (j - i) * hop / sr >= min_sil:
                cuts.append((i * hop, j * hop))
            i = j
        else:
            i += 1
    bnd = [0]
    for a, b in cuts:
        bnd += [a, b]
    bnd.append(len(x))
    segs = []
    for k in range(0, len(bnd) - 1, 2):
        s, e = bnd[k], bnd[k + 1]
        if min_len <= (e - s) / sr <= max_len:
            segs.append((s, e))
    return segs, cuts, hop


def f0_of(y, sr):
    f0, _, _ = librosa.pyin(y.astype(np.float64), fmin=60, fmax=600, sr=sr,
                            frame_length=2048, hop_length=256)
    if f0 is None:
        return None
    f = f0[~np.isnan(f0)]
    return f if len(f) >= 8 else None


def measure(p, name=""):
    x, sr = load(p)
    segs, cuts, hop = sent_windows(x, sr)
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

    W, H = int(0.05 * sr), int(0.025 * sr)
    n = (len(x) - W) // H
    e = np.sqrt(np.array([np.mean(x[i * H:i * H + W] ** 2) for i in range(n)]) + 1e-12)
    edb = 20 * np.log10(e + 1e-12)
    voiced = edb[edb > np.percentile(edb, 25)]
    C = float(np.percentile(voiced, 90) - np.percentile(voiced, 10)) if len(voiced) > 20 else None

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

    slopes = []
    for s, e in segs[:80]:
        a = max(s, e - int(0.35 * sr))
        if e - a < int(0.15 * sr):
            continue
        f = f0_of(x[a:e], sr)
        if f is None:
            continue
        ff = f[(f > np.percentile(f, 5)) & (f < np.percentile(f, 95))]
        if len(ff) < 5:
            continue
        semi = 12 * np.log2(ff / np.median(ff) + 1e-12)
        slopes.append(np.polyfit(np.arange(len(semi)), semi, 1)[0] * (sr / 256))
    E = float(np.std(slopes)) if len(slopes) >= 2 else None

    runs = np.array([(j - i) * hop / sr for i, j in cuts])
    runs = runs[runs >= 0.28]
    F = None
    if len(runs) >= 5:
        hist = np.histogram(runs, bins=8)[0].astype(float)
        prob = hist / hist.sum()
        prob = prob[prob > 0]
        F = float(-(prob * np.log2(prob)).sum() / np.log2(8))

    return dict(name=name or p.split("/")[-1], A=A, B=B, C=C, D=D, E=E, F=F,
                segs=len(segs), pauses=len(runs))


def _f(v, w, d):
    return ("%*.*f" % (w, d, v)) if v is not None else " " * (w - 1) + "-"


if __name__ == "__main__":
    print(f"{'样本':<26s} A=句内动态 B=全局音高 C=能量动态 D=语速CV E=句尾STD F=停顿熵")
    for p in sys.argv[1:]:
        try:
            m = measure(p)
            print(f"{m['name'][:26]:26s} A={_f(m['A'],5,2)} B={_f(m['B'],5,2)} "
                  f"C={_f(m['C'],5,2)} D={_f(m['D'],5,3)} E={_f(m['E'],5,2)} "
                  f"F={_f(m['F'],5,3)} | 句段={m['segs']} 停顿={m['pauses']}")
        except Exception as ex:
            print(f"{p.split('/')[-1][:26]:26s} ERR {repr(ex)[:100]}")
