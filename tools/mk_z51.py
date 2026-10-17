# -*- coding: utf-8 -*-
"""mk_z51.py —— z19 女声「句末短音」可闻性裁决素材。

## S6 的目的（🔴 规范硬要求）
物理上量到了差异，**不等于员工听得出来**（z44：13dB 改动，9 票仅 1 票指向修后）。
所以定根因前必须先出裁决题验证可闻性。本脚本只做素材，不出题。

## 现象（员工原话）
「xf5272-prod 说完…马上出来有个'嘎哈'的 0.2 秒音」

## 定位证据（能量包络目视，10ms 帧，全篇 mean=-31.3 dB）
- 20.22~20.28s（-23.1 dB）与 20.49~20.62s（-19.9 dB）：
  在「숨어있거든요.」句末（20.13 收干净）与「자」（21.08 起）之间的停顿里，**两个**短音
- 22.79~23.03s（峰值 -15.3 dB，**比全篇均值高 16 dB**）：
  在「하시고요.」句末（22.72 收干净）与「본격적으로」（23.50 起）之间

## A / B
- A = 现役原样（xf5272-prod-21）
- B = 把上面两段压到静音电平（-42 dB），边界 12ms raised-cosine 淡变防 click

🔴 压制后必须做 LUFS 对齐：否则员工听到 B 更轻就可能只凭响度选 A，
   裁决题就变成了"谁响"而不是"谁自然"（规范自检链的一环）。

用法：python3 mk_z51.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SRC = "/mnt/tts/output/xf5272-prod-21/xf5272-prod-1.mp3"
FB = "/mnt/tts/fb"
# 压制区间（秒）：(起点, 终点)
MUTE = [(20.15, 20.64), (22.77, 23.05)]
FADE = 0.012           # 淡变时长
TARGET_GAIN = 10 ** (-42.0 / 20)   # 压到 -42 dB
OUT_A = "z51_a_s1.wav"
OUT_B = "z51_b_s1.wav"


def load16k(p):
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ar", "16000", "-ac", "1", t], check=True)
    x, sr = sf.read(t, dtype="float32", always_2d=False)
    return x, sr


def load_keep(p):
    """保留原始采样率"""
    t = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                    "-ac", "1", t], check=True)
    x, sr = sf.read(t, dtype="float32", always_2d=False)
    return x, sr


def gain_curve(n, sr, mute, fade):
    g = np.ones(n, dtype=np.float64)
    fn = int(fade * sr)
    for a, b in mute:
        ia, ib = int(a * sr), int(b * sr)
        ia, ib = max(0, ia), min(n, ib)
        g[ia:ib] = TARGET_GAIN
        # 两端 raised-cosine 淡变
        for s, e, direction in ((ia - fn, ia, "in"), (ib, ib + fn, "out")):
            s, e = max(0, s), min(n, e)
            if e <= s:
                continue
            k = e - s
            w = 0.5 - 0.5 * np.cos(np.pi * np.arange(k) / k)
            if direction == "in":
                g[s:e] = 1.0 + (TARGET_GAIN - 1.0) * w
            else:
                g[s:e] = TARGET_GAIN + (1.0 - TARGET_GAIN) * w
    return g


def lufs(path):
    out = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", path, "-af", "ebur128=peak=true",
         "-f", "null", "-"], capture_output=True, text=True).stderr
    v = None
    for line in out.splitlines():
        if "I:" in line and "LUFS" in line:
            try:
                v = float(line.split("I:")[1].split("LUFS")[0].strip())
            except Exception:
                pass
    return v


def main():
    x, sr = load_keep(SRC)
    print("源：%s  %.1fs  %d Hz" % (SRC, len(x) / sr, sr))
    n = len(x)
    g = gain_curve(n, sr, MUTE, FADE)
    y = (x * g).astype(np.float32)

    pa = os.path.join(FB, OUT_A)
    pb = os.path.join(FB, OUT_B)
    sf.write(pa, x, sr)
    sf.write(pb, y, sr)
    print("A 原样 ->", pa)
    print("B 压制 ->", pb, "压制区间", MUTE)

    # 🔴 LUFS 对齐
    la, lb = lufs(pa), lufs(pb)
    print("LUFS  A=%.2f  B=%.2f" % (la, lb))
    if la is not None and lb is not None and abs(la - lb) > 0.05:
        d = la - lb
        y2 = (y * (10 ** (d / 20))).astype(np.float32)
        sf.write(pb, y2, sr)
        lb2 = lufs(pb)
        print("  已对齐：B 施加 %+.2f dB -> LUFS B=%.2f" % (d, lb2))
    # 完整性
    for p in (pa, pb):
        d = sf.info(p).duration
        print("  %s 时长 %.2fs  %s" % (os.path.basename(p), d,
                                       "OK" if abs(d - len(x) / sr) < 0.05 else "❌时长不一致"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
