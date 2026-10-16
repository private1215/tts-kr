# -*- coding: utf-8 -*-
"""第47轮：量化「波段杂音」= 8~16Hz 的周期性音量起伏。

## 为什么查这个
员工 z45 连着 4 题写「波段杂音」，第 1 题点名 2 号（GS 段）。
对比两条作业的生产日志，发现**两条根本不是同一条流水线**：

| | 现役 omnivoice | GS 微调 |
|---|---|---|
| 分块 | 1 块 | 2 块 |
| 包络整形 8~16Hz | **没跑** | **0.1301 -> 0.1511** |
| 慢尺度轮廓 | **没跑** | k_slow=1.148 |
| 时长对齐 atempo | **没跑** | 0.9308 / 0.9349 |
| 首尾静音 atempo | **没跑** | 0.931 |

omnivoice 属于 NEW_ENGINES → 走 `_legacy_post` 之外的**快路径**，
整条旧后处理链被绕过；gptsovits 走 legacy 全链，所以被
**注入了 8~16Hz 的包络整形** + **atempo 0.93 拉慢 7%**。
8~16Hz 的音量起伏 = 每秒抖 8~16 下，听感就是"嗡嗡/波段杂音"。

## 做法
5ms 帧 RMS → dB 包络 → 减 2s 滑动中位去掉句子级起伏 → FFT
→ 报 1~30Hz 里各频段的调制幅度（dB RMS），重点看 8~16Hz。
"""
import os
import subprocess
import sys

import numpy as np

SR = 24000
TMP = "/mnt/tts/diag/_mod47"
os.makedirs(TMP, exist_ok=True)
FB = "/mnt/tts/fb"
EL = "/mnt/tts/data/cases/00b6a45ad6aa/merged.wav"

MP3S = [
    ("A_现役 s1", os.path.join(FB, "z45_a_s1.mp3")),
    ("B_GS   s1", os.path.join(FB, "z45_b_s1.mp3")),
    ("A_现役 s2", os.path.join(FB, "z45_a_s2.mp3")),
    ("B_GS   s2", os.path.join(FB, "z45_b_s2.mp3")),
    ("A_现役 s3", os.path.join(FB, "z45_a_s3.mp3")),
    ("B_GS   s3", os.path.join(FB, "z45_b_s3.mp3")),
    ("A_现役 el", os.path.join(FB, "z45_a_el.mp3")),
    ("B_EL   el", os.path.join(FB, "z45_b_el.mp3")),
]


def wav(src, tag):
    o = os.path.join(TMP, "%s.wav" % tag.replace(" ", "_"))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", o],
                   check=True)
    import soundfile as sf
    return sf.read(o, dtype="float32")[0]


def env_db(y, ms=5.0):
    n = int(SR * ms / 1000.0)
    k = len(y) // n
    F = y[:k * n].reshape(k, n)
    r = np.sqrt((F ** 2).mean(axis=1))
    return 20 * np.log10(np.maximum(r, 1e-9)), 1000.0 / ms


def band_db(e, fs_hz, lo, hi):
    """e 已去趋势。返回该频段的调制幅度（dB RMS）。"""
    x = e - e.mean()
    w = np.hanning(len(x))
    S = np.abs(np.fft.rfft(x * w)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs_hz)
    m = (f >= lo) & (f < hi)
    if not m.any():
        return 0.0
    # 功率谱 -> 该频段总功率 -> 幅度（用 sqrt(2*P)/N 近似 RMS）
    P = S[m].sum()
    rms = np.sqrt(2.0 * P) / max(len(x), 1)
    return float(rms)


def desc(name, y):
    e, fs_hz = env_db(y)
    # 去句子级趋势：2s 滑动中位
    win = int(2.0 * fs_hz)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    ep = np.pad(e, pad, mode="edge")
    trend = np.array([np.median(ep[i:i + win]) for i in range(len(e))])
    d = e - trend
    return (name,
            band_db(d, fs_hz, 1, 4),
            band_db(d, fs_hz, 4, 8),
            band_db(d, fs_hz, 8, 16),
            band_db(d, fs_hz, 16, 30),
            float(np.std(d)))


def main():
    rows = [desc(nm, wav(p, nm)) for nm, p in MP3S]
    rows.append(desc("RAW_EL  s1", wav(
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "12", "-t", "20",
                        "-i", EL, "-ac", "1", "-ar", str(SR),
                        "-c:a", "pcm_s16le",
                        os.path.join(TMP, "raw_el.wav")], check=True) and
        os.path.join(TMP, "raw_el.wav"), "RAW_EL")))
    print("%-12s %9s %9s %9s %9s %9s"
          % ("样本", "1-4Hz", "4-8Hz", "8-16Hz", "16-30Hz", "总残余"))
    for r in rows:
        print("%-12s %9.3f %9.3f %9.3f %9.3f %9.3f" % r)
    print("\n（单位 dB RMS；8~16Hz 那一列就是「包络整形」注入的频段）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
