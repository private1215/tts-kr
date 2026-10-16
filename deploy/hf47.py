#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hf47.py —— 排除「亮度混淆」：同一带宽下重算 click 密度。

背景：loc47/clickctx47 量到现役 click 密度 0.48%、EL 0.26%、GS 0.01%。
但 |Δ|（相邻样本跳变）天然对高频敏感 —— 谁高频多，谁的 |Δ| 就大。
GS 已知能量 83% 堆在 300~1000Hz（比现役闷得多），click 少可能只是**暗**。

本脚本做两件事：
  ① 报告各素材的频段能量占比（亮度）
  ② 把三方**低通到同一截止频率**后重算 click 密度 —— 同带宽才有可比性

若低通后三方 click 密度趋同 → |Δ| 差异只是亮度，不是缺陷，不用修。
若低通后现役仍显著高   → 真缺陷，且定位在发声段的波形不连续。
"""
import subprocess

import numpy as np

SR = 16000


def load(path, extra=None):
    cmd = ["ffmpeg", "-v", "error", "-i", path]
    if extra:
        cmd += extra
    cmd += ["-ar", str(SR), "-ac", "1", "-f", "f32le", "-"]
    p = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32)


def db(v):
    return 20.0 * np.log10(max(v, 1e-12))


def band_share(x, sr=SR):
    """各频段能量占比（%）"""
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    tot = X.sum()
    bands = [(0, 300), (300, 1000), (1000, 3000), (3000, 5000),
             (5000, 8000), (8000, 16000)]
    out = []
    for lo, hi in bands:
        m = (f >= lo) & (f < hi)
        out.append((lo, hi, float(X[m].sum() / tot * 100)))
    return out


def click_density(x, sr=SR, thr=0.30):
    n = int(sr * 0.02)
    d = np.abs(np.diff(x))
    m = (len(d) // n) * n
    dmax = d[:m].reshape(-1, n).max(axis=1)
    return len(np.where(dmax >= thr)[0]), len(dmax), float(dmax.mean())


def main():
    srcs = [
        ("EL 原声 wav", "/mnt/tts/data/cases/00b6a45ad6aa/merged.wav"),
        ("EL 压 mp3 192k", "/mnt/tts/diag/loc47/el_192k.mp3"),
        ("现役 omnivoice",
         "/mnt/tts/output/sola47_f_omnivoice/sola47_f_omnivoice-1.mp3"),
        ("GS 微调 块1",
         "/mnt/tts/output/sola47_f_gptsovits-2/sola47_f_gptsovits-1.mp3"),
    ]

    print("\n########## ① 频段能量占比 (%) ##########")
    print(f"{'素材':<18}" + "".join(f"{lo//1000 if lo>=1000 else lo}k-"
                                    f"{hi//1000 if hi>=1000 else hi}k".rjust(9)
                                    for lo, hi in
                                    [(0, 300), (300, 1000), (1000, 3000),
                                     (3000, 5000), (5000, 8000),
                                     (8000, 16000)]))
    for tag, p in srcs:
        x = load(p)
        bs = band_share(x)
        print(f"{tag:<18}" + "".join(f"{v:9.1f}" for _, _, v in bs))

    for lp in (8000, 5000):
        print(f"\n########## ② 统一低通 {lp}Hz 后的 click 密度 ##########")
        print(f"{'素材':<18}{'|Δ|>=0.3 窗':>12}{'占比':>9}{'|Δ| 均值':>10}")
        for tag, p in srcs:
            x = load(p, ["-af", f"lowpass=f={lp}:poles=2"])
            k, tot, mean = click_density(x)
            print(f"{tag:<18}{k:>12}{k/tot*100:>8.2f}%{mean:>10.4f}")

    print("\n########## ③ 参考：各自原始带宽的 click 密度 ##########")
    print(f"{'素材':<18}{'|Δ|>=0.3 窗':>12}{'占比':>9}{'|Δ| 均值':>10}")
    for tag, p in srcs:
        x = load(p)
        k, tot, mean = click_density(x)
        print(f"{tag:<18}{k:>12}{k/tot*100:>8.2f}%{mean:>10.4f}")


if __name__ == "__main__":
    main()
