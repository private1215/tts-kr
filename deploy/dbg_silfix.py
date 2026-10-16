#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为什么 95.78s 那段填了还是 -70dB？逐步打调试。"""
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, "/mnt/tts/diag")
SR = 24000
HOP = 0.005
TMP = "/mnt/tts/diag/_silfix"
SRC = "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"
THR, TGT = -60.0, -52.0


def load(src, tag):
    o = os.path.join(TMP, "_d%s.wav" % tag)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ac", "1",
                    "-ar", str(SR), o], check=True)
    return sf.read(o, dtype="float32")[0]


def env_db(y):
    h = int(HOP * SR)
    n = len(y) // h
    e = np.sqrt(np.array([(y[i * h:(i + 1) * h] ** 2).mean() for i in range(n)]))
    return 20 * np.log10(e + 1e-12)


y = load(SRC, "src")
db = env_db(y)
print("总帧 %d  时长 %.1fs" % (len(db), len(y) / SR))

# 找出含 95.78s 的 <THR 连续段
target_f = int(95.78 / HOP)
m = db < THR
s = target_f
while s > 0 and m[s - 1]:
    s -= 1
e = target_f
while e < len(m) - 1 and m[e + 1]:
    e += 1
print("95.78s 所在段：帧 %d~%d = %.3f~%.3fs  时长 %.0fms  帧内最低 %.1fdB"
      % (s, e, s * HOP, e * HOP, (e - s) * HOP * 1000, db[s:e + 1].min()))
print("  minlen 门槛：需 >= 40ms（%.0fms %s）" % ((e - s) * HOP * 1000,
                                               "过" if (e - s) * HOP * 1000 >= 40 else "不过"))

# 全篇 <THR 段统计
d = np.diff(m.astype(np.int8))
st = list(np.where(d == 1)[0] + 1)
en = list(np.where(d == -1)[0] + 1)
if m[0]:
    st.insert(0, 0)
if m[-1]:
    en.append(len(m))
allsegs = list(zip(st, en))
ok = [(a, b) for a, b in allsegs if (b - a) * HOP * 1000 >= 40]
print("全篇 <THR 段 %d 个，其中 >=40ms 的 %d 个" % (len(allsegs), len(ok)))
print("  含 95.78 的段在 ok 列表里？",
      any(abs(a - s) < 3 or abs(b - e) < 3 for a, b in ok))

amp = 10.0 ** (TGT / 20.0)
print("\n目标幅度 amp=%.6f  keep 阈值 amp*1.5=%.6f" % (amp, amp * 1.5))
a, b = int(s * HOP * SR), int(e * HOP * SR)
sy = y[a:b]
print("段内样本 %d：RMS %.6f(%.1fdB)  峰值 %.6f  超过 keep 阈值的样本占比 %.2f%%"
      % (len(sy), np.sqrt((sy ** 2).mean()),
         20 * np.log10(np.sqrt((sy ** 2).mean()) + 1e-12),
         np.abs(sy).max(), 100.0 * (np.abs(sy) > amp * 1.5).mean()))

# roomtone
band = (db > TGT - 4) & (db < TGT + 6)
db2 = np.diff(band.astype(np.int8))
st2 = list(np.where(db2 == 1)[0] + 1)
en2 = list(np.where(db2 == -1)[0] + 1)
if band[0]:
    st2.insert(0, 0)
if band[-1]:
    en2.append(len(band))
c = [(x, z) for x, z in zip(st2, en2) if (z - x) * HOP >= 0.25]
print("\n房间底噪候选(-56~-46dB, >=250ms) %d 个" % len(c))
if c:
    x, z = max(c, key=lambda t: t[1] - t[0])
    rt = y[int(x * HOP * SR):int(z * HOP * SR)]
    rt_rms = float(np.sqrt((rt ** 2).mean()))
    print("  选中 %.2f~%.2fs 长 %.2fs  RMS %.6f(%.1fdB)"
          % (x * HOP, z * HOP, (z - x) * HOP, rt_rms, 20 * np.log10(rt_rms)))
    print("  放大倍率 amp/rt_rms = %.3f" % (amp / rt_rms))
else:
    print("  ❌ 没有候选 —— 会退回 pink 分支")
