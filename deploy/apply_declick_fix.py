#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第八个方案（也是找到了真根因）：修 declick 的**过零漏洞**

## 🔴🔴 根因（zchk.py 实测，同文案 EL 逐秒对照）
| 文件 | 最小电平 | <−100dB 帧数 |
|---|---|---|
| 我们 裸推理 | **−240dB** | 343 |
| 我们 +后处理 | **−240dB** | 159 |
| 我们 反馈台 mp3 | **−240dB** | 32 |
| **EL 原声** | **−98.5dB** | **0** |

EL **从来没有数字零**，我们几百帧。而且 wav→mp3→wav 实测 13 帧不变
→ **mp3 不是元凶**，数字零来自生成/后处理链路本身。

## bug 在哪
`app/declick.py` 的填充是**逐样本取二者较大**：
    out = where(|噪声| > |原信号|, 噪声, 原信号)
但**粉噪会过零** —— 噪声样本幅度接近 0 时条件不成立，于是数字零**原样保留**。
这就是为什么 declick「跑了」却还有 32~159 帧数字零。

## 为什么它正好解释员工所有反馈
- 「断句有异声」：断口都在句间 ✅
- 「突然多出一个短音」：−240dB 的绝对静音在听觉上会**凸显紧邻的音节** ✅
- 「音量大小有变化」：−240dB → −20dB 是 220dB 的落差 ✅
- EL 没有这个毛病：EL 最安静只到 −98.5dB ✅

## 修法
对**每个样本**：若 |x| < −100dB，直接替换为粉噪（电平 −89dB，对齐 DEClick 的
原设计目标 / EL 深静音）。替换后不存在连续静音；粉噪即使过零也是**连续波形**的一部分，
不会再被听成断口。
⚠️ 时长严格不变。

## 与之前七次失败的区别
前七次改的都是**可听内容**（拖尾、底噪、起音、顶起幅度），所以要么无效要么引入副作用。
这一次改的是**不可听的绝对静音**（<−100dBFS，人耳阈值远高于此），
按定义不可能改变任何"内容"，只消除断口。

用法：/mnt/tts/projects/qwen3-tts/venv/bin/python apply_declick_fix.py
"""
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

SR = 24000
HOP = 0.005
TMP = "/mnt/tts/diag/_dcfix"
os.makedirs(TMP, exist_ok=True)
SRC = "/mnt/tts/diag/xf5272/prod_EL_female_52727ec43154.mp3"
MARKS = [95.5, 99.0, 112.0, 131.9, 170.5]


def load(src, tag="x", sr=SR):
    o = os.path.join(TMP, "%s.wav" % tag)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ac", "1",
                    "-ar", str(sr), o], check=True)
    return sf.read(o, dtype="float32")[0], sr


def env_db(y, sr):
    hop = int(HOP * sr)
    n = len(y) // hop
    e = np.sqrt(np.array([(y[i * hop:(i + 1) * hop] ** 2).mean() for i in range(n)]))
    return 20 * np.log10(e + 1e-12)


def pink(n, rng):
    w = rng.standard_normal(n)
    b = np.convolve(w, np.ones(64) / 64.0, mode="same")
    return b / (np.abs(b).max() + 1e-12) * 0.6


def proc(src, dst, quiet_db=-100.0, fill_db=-89.0):
    y, sr = load(src, "src")
    rng = np.random.default_rng(20261016)
    thr = 10.0 ** (quiet_db / 20.0)
    amp = 10.0 ** (fill_db / 20.0)
    m = np.abs(y) < thr
    n_fix = int(m.sum())
    if n_fix:
        nz = pink(len(y), rng) * amp
        y = np.where(m, nz, y)          # 🔴 逐样本替换，不再"取较大"
    pk = float(np.abs(y).max())
    if pk > 0.99:
        y = y * (0.99 / pk)
    sf.write(os.path.join(TMP, "o.wav"), y, sr)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", os.path.join(TMP, "o.wav"),
                    "-b:a", "192k", dst], check=True)
    return n_fix


def report(path, tag, sr=SR):
    y, _ = load(path, tag[:5])
    db = env_db(y, sr)
    print("  %-22s 最小 %7.1fdB   <−100dB帧 %4d   <−90dB帧 %4d"
          % (tag, db.min(), int((db < -100).sum()), int((db < -90).sum())))
    for t in MARKS:
        a = int((t - 0.3) / HOP)
        b = int((t + 0.5) / HOP)
        seg = db[a:b]
        print("      %.1fs 最低 %.1fdB" % (t, seg.min()))


def main():
    print("=== 修前 ===")
    report(SRC, "现役成品")
    for fd in (-89.0, -96.0):
        dst = os.path.join(TMP, "fix_%g.mp3" % abs(fd))
        nf = proc(SRC, dst, -100.0, fd)
        print("\n=== 修后（填充 %.0fdB，替换 %d 个样本）===" % (fd, nf))
        report(dst, "fill%g" % fd)
    print("\n参考：EL 原声最小值 −98.5dB、数字零 0 帧")
    print("\nDCFIX_DONE")


if __name__ == "__main__":
    main()
