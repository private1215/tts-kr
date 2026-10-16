#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""peaknorm47.py —— 电平归一后再判 click，堵掉「绝对阈值」的口径漏洞。

教训（本轮自己踩的坑）：
  |Δ| >= 0.30 是**绝对**阈值，对整体电平极度敏感。实测同一条素材
  24k->16k 重采样后 click 0.27%（峰值 0.7196），仅仅 +1.4dB 静态增益
  就变成 0.71%（峰值 0.8455）、loudnorm 后 6.29%（峰值 0.8499）。
  → 不归一电平就横向比，等于在比「谁声音大」，不是在比「谁有咔哒声」。

本脚本把每个素材**峰值统一归一到 PEAK**，再算 click 密度。

同时也给第二个、对电平更鲁棒的指标：
  crest_per_win = max|x| / rms  的异常窗比例
  —— 瞬态尖峰相对本窗能量的比值，增益对它无效。
"""
import subprocess

import numpy as np

SR = 16000
PEAK = 0.70


def load(path):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path,
                        "-ar", str(SR), "-ac", "1", "-f", "f32le", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32)


def norm_peak(x, peak=PEAK):
    m = np.abs(x).max()
    return x * (peak / m) if m > 1e-9 else x


def click(x, thr=0.30):
    """归一后 click 密度"""
    n = int(SR * 0.02)
    d = np.abs(np.diff(x))
    m = (len(d) // n) * n
    dmax = d[:m].reshape(-1, n).max(axis=1)
    return int((dmax >= thr).sum()), len(dmax), float(dmax.mean())


def spike_ratio(x, k=6.0):
    """对电平鲁棒的尖峰指标：窗内 max|x| 超过该窗 rms 的 k 倍。
    增益对 max/rms 无效（同比例缩放），所以这才是真·咔哒声探针。"""
    n = int(SR * 0.02)
    m = (len(x) // n) * n
    fr = x[:m].reshape(-1, n)
    mx = np.abs(fr).max(axis=1)
    rm = np.sqrt((fr ** 2).mean(axis=1)) + 1e-9
    cr = mx / rm
    thr = np.median(cr) * k
    return int((cr >= thr).sum()), len(cr), float(np.median(cr))


SRC = [
    ("EL 原声 wav(24k)", "/mnt/tts/data/cases/00b6a45ad6aa/merged.wav"),
    ("现役 omnivoice", "/mnt/tts/output/sola47_f_omnivoice/"
                       "sola47_f_omnivoice-1.mp3"),
    ("GS 微调 块1", "/mnt/tts/output/sola47_f_gptsovits-2/"
                    "sola47_f_gptsovits-1.mp3"),
    ("引擎裸输出 step0", "/mnt/tts/diag/raw47/step0_raw.wav"),
    ("fix_chain 后 step1", "/mnt/tts/diag/raw47/step1.wav"),
    ("裸+重采样 r_only", "/mnt/tts/diag/raw47/r_only.wav"),
    ("裸+重采样+1.4dB", "/mnt/tts/diag/raw47/r_vol.wav"),
    ("裸+loudnorm(动态)", "/mnt/tts/diag/raw47/r_loud.wav"),
]


def main():
    print(f"全部归一到峰值 {PEAK}，再判 click\n")
    print(f"{'素材':<22}{'click窗':>8}{'占比':>9}{'|Δ|均':>9}"
          f"{'尖峰窗':>8}{'占比':>9}{'crest中位':>10}")
    print("-" * 76)
    for tag, p in SRC:
        x = norm_peak(load(p))
        k, tot, mean = click(x)
        sk, stot, cmed = spike_ratio(x)
        print(f"{tag:<22}{k:>8}{k/tot*100:>8.2f}%{mean:>9.4f}"
              f"{sk:>8}{sk/stot*100:>8.2f}%{cmed:>10.2f}")


if __name__ == "__main__":
    main()
