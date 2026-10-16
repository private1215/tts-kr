# -*- coding: utf-8 -*-
"""第47轮：波形自相似 —— 直接查「重复念字」，不依赖 ASR。

## 为什么不用 ASR
whisper 解码时很可能把结巴/重复**归一化掉**（语言模型倾向于输出通顺文本），
所以「ASR 没检出重复」不能证明音频里没有重复。
必须在**波形/频谱层面**做自相似：找间隔 0.2~3.0s 的两段音频是否几乎一样。

## 做法
- 20ms 帧、10ms hop 的梅尔谱 → 每帧一个向量
- 对 lag = 0.2~3.0s 的每一个候选，算「帧 i 与帧 i+lag 的余弦相似度」的中位
- 若某个 lag 上的中位相似度异常高（> 0.9），说明整段被**平移重复**了
- 同时报「最相似的连续片段对」（长度 >= 0.4s）
"""
import subprocess
import sys

import numpy as np

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")

CUR = "/mnt/tts/diag/sola47_f_omnivoice.wav"
GS = "/mnt/tts/diag/sola47_f_gptsovits.wav"
DUR = 20.0
SR = 16000
FR = 320          # 20ms
HOP = 160         # 10ms


def mel(src, t0, dur, tag):
    o = "/mnt/tts/diag/_ss_%s.wav" % tag
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-ss", "%.2f" % t0, "-t", "%.1f" % dur,
                    "-ac", "1", "-ar", str(SR), o], check=True)
    import soundfile as sf
    y = sf.read(o, dtype="float32")[0]
    n = len(y)
    k = (n - FR) // HOP + 1
    idx = np.arange(FR)[None, :] + HOP * np.arange(k)[:, None]
    frames = y[idx] * np.hanning(FR)[None, :]
    S = np.abs(np.fft.rfft(frames, n=512, axis=1))
    # 对数压缩 + 去均值
    L = np.log(S + 1e-6)
    L = L - L.mean(axis=1, keepdims=True)
    nv = np.linalg.norm(L, axis=1, keepdims=True)
    return L / np.maximum(nv, 1e-9)


def scan(tag, src, t0):
    V = mel(src, t0, DUR, tag)
    k = len(V)
    fps = SR / float(HOP)
    lo, hi = int(0.2 * fps), int(3.0 * fps)
    best = []
    for lag in range(lo, hi):
        if lag >= k:
            break
        a, b = V[:k - lag], V[lag:]
        c = (a * b).sum(axis=1)
        best.append((float(np.median(c)), lag / fps))
    best.sort(reverse=True)
    print("=== %s（%s 第 %.2fs 起）===" % (tag, src.split("/")[-1], t0))
    print("  lag 0.2~3.0s 里中位相似度最高的 5 个：")
    for m, lag in best[:5]:
        print("     lag %.2fs  中位余弦 %.4f" % (lag, m))
    # 找最相似的连续片段（>= 0.4s）
    top = best[0]
    lag_n = int(round(top[1] * fps))
    a, b = V[:k - lag_n], V[lag_n:]
    c = (a * b).sum(axis=1)
    w = int(0.4 * fps)
    if len(c) >= w:
        cs = np.convolve(c, np.ones(w) / w, mode="valid")
        i = int(np.argmax(cs))
        print("  最相似的连续 %.1fs 片段：第 %.2fs ↔ 第 %.2fs，平均余弦 %.4f"
              % (w / fps, i / fps, (i + lag_n) / fps, cs[i]))
        if cs[i] > 0.90:
            print("  🔴 疑似重复念字（>= 0.90）")
        else:
            print("  ✅ 未见重复（阈值 0.90）")
    print()
    return best[0]


def main():
    scan("A_现役_s3", CUR, 319.79)
    scan("B_GS_s3", GS, 313.90)
    scan("A_现役_s1", CUR, 11.99)
    scan("B_GS_s1", GS, 11.77)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
