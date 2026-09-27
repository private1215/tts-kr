# -*- coding: utf-8 -*-
"""诊断已生成音频：按时间窗统计有声比例 / 静音分布，定位语速异常区间。"""
import sys
import numpy as np
import soundfile as sf

path = sys.argv[1]
x, sr = sf.read(path, dtype="float32")
if x.ndim > 1:
    x = x.mean(axis=1)
dur = len(x) / sr
print(f"文件 {path}  时长 {dur:.2f}s  sr={sr}")

# ---- 1. 逐 2 秒窗口：RMS / 有声比例 ----
win = int(2.0 * sr)
print("\n=== 每 2 秒窗口 ===")
print("时间(s)   RMS(dB)   有声比例%")
for i in range(0, len(x), win):
    seg = x[i:i + win]
    if len(seg) < sr * 0.2:
        break
    rms = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9)
    # 有声比例：短时能量超过全局能量 5% 的帧占比
    fr = int(0.02 * sr)
    n = len(seg) // fr
    if n == 0:
        continue
    e = np.array([np.sqrt(np.mean(seg[j * fr:(j + 1) * fr] ** 2)) for j in range(n)])
    g = np.sqrt(np.mean(x ** 2))
    voiced = float(np.mean(e > g * 0.08)) * 100
    t0 = i / sr
    bar = "#" * int(voiced / 4)
    print(f"{t0:6.1f}  {rms:8.1f}   {voiced:6.1f}  {bar}")

# ---- 2. 所有静音段（更低阈值）----
print("\n=== 静音段 (>350ms, 阈值 -45dB) ===")
fr = int(0.02 * sr)
n = len(x) // fr
e = np.array([np.sqrt(np.mean(x[j * fr:(j + 1) * fr] ** 2)) for j in range(n)])
thr = 10 ** (-45 / 20)
sil = e < thr
runs = []
cur = None
for j, v in enumerate(sil):
    if v and cur is None:
        cur = j
    elif not v and cur is not None:
        runs.append((cur * fr / sr, j * fr / sr))
        cur = None
if cur is not None:
    runs.append((cur * fr / sr, n * fr / sr))
for a, b in runs:
    if b - a >= 0.35:
        print(f"  {a:7.2f}s -> {b:7.2f}s   时长 {b-a:.2f}s")
tot_sil = sum(b - a for a, b in runs if b - a >= 0.35)
print(f"  静音合计 {tot_sil:.2f}s ({tot_sil/dur*100:.1f}%)")

# ---- 3. 尾部检查：是否常数拖尾 ----
print("\n=== 尾部 1 秒是否常数（变速 clip bug 特征）===")
tail = x[-sr:]
print(f"  最后 1s 唯一值数量: {len(np.unique(np.round(tail, 5)))}  (很小=常数拖尾)")
print(f"  最后 1s RMS: {20*np.log10(np.sqrt(np.mean(tail**2))+1e-9):.1f} dB")
