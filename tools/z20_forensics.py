# -*- coding: utf-8 -*-
"""z20_forensics.py —— 把「为什么 GS 听着像机器」拆成图谱和数据。

## 为什么要做这个

z20 盲听结果是：GS 男女声都拿 0 分（明显是机器），EL 满分 2 分。
但 GS 的**常规客观指标全面优于 EL**：
    click 0.172/s（EL 的 0.6 倍）、音色 0.9196、杂音 4.121、CER 0.0527

用户质疑（2026-10-17）：
> 「案例里的声音拆成图谱和数据，按道理来说，只要吐字发音标准，应该不会有
>   问题啊，为什么转折那么大？」

这个质疑是对的：**CER 只衡量「念对了没有」，完全不衡量「像不像人」**。
常规那几个指标（click / 音色 / 杂音）也是体检项，不是验收项。

所以要回答「机器感从哪来」，必须看**真正决定听感**的物理量：
- LTAS（长时平均频谱）：频响形状，金属声 / 发闷都写在这里
- 频带能量占比：低频厚不厚、高频亮不亮、有没有齿音爆炸
- 频谱质心 / 滚降 / 带宽：整体音色亮度与集中度
- **F0 统计（均值 / 标准差 / 范围）**：语调起伏。真人说话 F0 一直在动，
  合成语音常常「一条直线」，这是「机器感」最常见的来源
- **谐波噪声比 HNR**：声码器合成的语音周期性异常（过于纯净 = 电子音，
  噪声过多 = 沙哑），是机器感的核心指标
- 频谱平坦度：噪声 vs 乐音的占比
- 调制谱：能量起伏的节奏（音节率附近）

## 用法

    python3 z20_forensics.py <片段目录> <输出目录>

只跑本地，不碰服务器。
"""
import json
import os
import sys

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SR = 16000
# 频道划分（Hz）—— 对应听感
BANDS = [
    ("低频 0-300", 0, 300),          # 厚度 / 底气
    ("中低 300-1k", 300, 1000),      # 基频与主要共振峰
    ("中频 1k-3k", 1000, 3000),      # 清晰度 / 临场感
    ("高频 3k-6k", 3000, 6000),      # 齿音 / 明亮度
    ("超高频 6k-8k", 6000, 8000),    # 嘶声 / 空气感
]


def load(path, sr=SR):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def ltas(y, sr=SR, n_fft=1024):
    """长时平均频谱（只统计有声帧，避免静音把谱拉平）。"""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=256)) ** 2
    db = librosa.power_to_db(S, ref=np.max)
    frame_db = db.mean(axis=0)
    voiced = frame_db > (np.percentile(frame_db, 30))
    if voiced.sum() < 10:
        voiced = np.ones_like(frame_db, dtype=bool)
    return db[:, voiced].mean(axis=1), librosa.fft_frequencies(sr=sr, n_fft=n_fft)


def band_share(y, sr=SR, n_fft=1024):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=256)) ** 2
    db = librosa.power_to_db(S, ref=np.max)
    frame_db = db.mean(axis=0)
    voiced = frame_db > np.percentile(frame_db, 30)
    P = (S[:, voiced].mean(axis=1) if voiced.sum() > 10 else S.mean(axis=1))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    total = P.sum() + 1e-12
    return {name: float(P[(freqs >= lo) & (freqs < hi)].sum() / total * 100)
            for name, lo, hi in BANDS}


def f0_stats(y, sr=SR):
    """F0 均值/标准差/范围 —— 语调起伏。真人 std 大，机器常常很小。"""
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, sr=sr, fmin=60, fmax=500, frame_length=1024, hop_length=256)
    f0v = f0[voiced_flag & np.isfinite(f0)] if f0 is not None else np.array([])
    if len(f0v) < 20:
        return {"n": int(len(f0v)), "mean": None, "std": None,
                "p10": None, "p90": None, "range_ratio": None, "vprob": None}
    return {
        "n": int(len(f0v)),
        "mean": float(np.mean(f0v)),
        "std": float(np.std(f0v)),
        "p10": float(np.percentile(f0v, 10)),
        "p90": float(np.percentile(f0v, 90)),
        # 🔴 相对起伏：std/mean。真人说话通常在 0.15~0.35，机器常常 <0.12
        "range_ratio": float(np.std(f0v) / np.mean(f0v)),
        "vprob": float(np.nanmean(voiced_prob)) if voiced_prob is not None else None,
    }


def spectral_feats(y, sr=SR):
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    cent = librosa.feature.spectral_centroid(S=S, sr=sr)
    roll = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)
    bw = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    flat = librosa.feature.spectral_flatness(S=S)
    return {
        "centroid_hz": float(np.median(cent)),
        "rolloff85_hz": float(np.median(roll)),
        "bandwidth_hz": float(np.median(bw)),
        "flatness_med": float(np.median(flat)),
    }


def hnr_approx(y, sr=SR, n_fft=1024):
    """近似谐波噪声比：用谐波/打击分离的能量比。

    值高 = 周期性主导（纯净，可能「电子音」）；值低 = 噪声多（沙哑）。
    """
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=256)) ** 2
    H, P = librosa.decompose.hpss(S, margin=2.0)
    h = H.sum() + 1e-12
    p = P.sum() + 1e-12
    return {"hnr_db": float(10 * np.log10(h / p)),
            "harmonic_share_pct": float(h / (h + p) * 100)}


def modulation(y, sr=SR):
    """能量包络在 2~10 Hz（音节率附近）的调制深度 —— 语调起伏的节奏。"""
    env = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    env = env - env.mean()
    if len(env) < 64:
        return {"mod_2_10_ratio": None}
    fps = sr / 256.0
    spec = np.abs(np.fft.rfft(env))
    freqs = np.fft.rfftfreq(len(env), d=1.0 / fps)
    band = (freqs >= 2) & (freqs <= 10)
    total = spec.sum() + 1e-12
    return {"mod_2_10_ratio": float(spec[band].sum() / total)}


def analyze(path):
    y = load(path)
    out = {"name": os.path.basename(path), "dur_s": float(len(y) / SR)}
    out.update(spectral_feats(y))
    out.update(hnr_approx(y))
    out.update(modulation(y))
    out["bands_pct"] = band_share(y)
    out["f0"] = f0_stats(y)
    lt, fr = ltas(y)
    out["_ltas"] = lt.tolist()
    out["_freqs"] = fr.tolist()
    return out


def main():
    indir, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    files = sorted(f for f in os.listdir(indir)
                   if f.endswith((".wav", ".mp3")))
    res = {}
    for f in files:
        p = os.path.join(indir, f)
        print("分析", f, flush=True)
        res[f] = analyze(p)
    json.dump(res, open(os.path.join(outdir, "forensics.json"), "w"),
              ensure_ascii=False, indent=1)

    # ---- 打印对照表 ----
    keys = list(res)
    print("\n%-16s %8s %8s %8s %9s %8s" %
          ("文件", "质心Hz", "滚降Hz", "带宽Hz", "HNR(dB)", "F0相对起伏"))
    for k in keys:
        r = res[k]
        f0 = r["f0"]
        print("%-16s %8.0f %8.0f %8.0f %9.2f %8s" % (
            k[:16], r["centroid_hz"], r["rolloff85_hz"], r["bandwidth_hz"],
            r["hnr_db"],
            ("%.3f" % f0["range_ratio"]) if f0.get("range_ratio") else "n/a"))

    print("\n%-16s %s" % ("文件", "频带能量占比 %"))
    hdr = "  ".join("%10s" % b[0] for b in BANDS)
    print("%-16s %s" % ("", hdr))
    for k in keys:
        b = res[k]["bands_pct"]
        print("%-16s %s" % (k[:16],
                            "  ".join("%10.1f" % b[n] for n, _, _ in BANDS)))

    print("\n-> %s/forensics.json" % outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
