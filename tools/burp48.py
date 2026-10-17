# -*- coding: utf-8 -*-
"""burp48.py —— 找「说完一个词，马上冒出来的短音」（员工说的"嘎哈"）。

## 与 gapnoise48.py 的区别
gapnoise48 找的是**夹在两处长静音中间**的孤立短音，结果 0 处 ——
因为员工说的是「说完 … **马上**出来」，这个音**紧贴发声**，
前后根本没有 0.15s 的静音，被那个判据滤掉了。

所以这里换判据：不要求前后静音，只找**能量突起** ——
某小段的电平比它周围（前 0.5s + 后 0.5s）的中位电平高出 DB_UP 以上，
且持续时长落在 MIN~MAX 之间。

## 🔴 必须先排除 whisper 幻觉（第48轮刚踩过）
ASR 的 insert **不能**证明"多念了音"：实测 xf5272-prod 的 28.36s 处
ASR 报 insert "와"，但那段电平比句间静音还低 17dB —— 是底噪，whisper 脑补的。
→ 本脚本每个候选都算出**相对全篇 RMS** 和**相对局部基线**的电平，
  只有明显高于局部底噪的才算真突起。

用法： python burp48.py <音频> [更多...]
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 0.005               # 5ms 细帧
MIN_LEN, MAX_LEN = 0.05, 0.45
DB_UP = 6.0                 # 高出局部基线多少 dB 才算突起
MIN_DB_REL = -25.0          # 相对全篇 RMS 不能太弱（否则是底噪）


def load(p):
    if not p.endswith(".wav"):
        t = tempfile.mktemp(suffix=".wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                        "-ar", str(SR), "-ac", "1", t], check=True)
        p = t
    x, sr = sf.read(p, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def db(v):
    # 🔴 必须用 np.maximum：输入是 ndarray 时内置 max() 会抛
    #    "truth value of an array ... is ambiguous"
    return 20 * np.log10(np.maximum(v, 1e-12))


def main():
    if len(sys.argv) < 2:
        print("用法: burp48.py <音频...>")
        return 1
    for p in sys.argv[1:]:
        x, sr = load(p)
        n = int(sr * FRAME)
        m = (len(x) // n) * n
        fr = x[:m].reshape(-1, n)
        rms = np.sqrt((fr ** 2).mean(axis=1))
        full = float(np.sqrt((x ** 2).mean()))
        win = int(0.5 / FRAME)          # 局部基线窗口 ±0.5s

        # 局部基线：前后各 0.5s 的 RMS 中位数（不含自己）
        base = np.empty_like(rms)
        for i in range(len(rms)):
            a = max(0, i - win)
            b = min(len(rms), i + win + 1)
            ctx = np.concatenate([rms[a:i], rms[i + 1:b]]) if i > a else rms[i + 1:b]
            base[i] = np.median(ctx) if len(ctx) else rms[i]

        up = db(rms) - db(base)
        cand = up >= DB_UP
        # 连续段
        segs, i = [], 0
        while i < len(cand):
            if cand[i]:
                j = i
                while j + 1 < len(cand) and cand[j + 1]:
                    j += 1
                segs.append((i, j))
                i = j + 1
            else:
                i += 1

        print("\n=== %s（%.1fs）===" % (os.path.basename(p), m / sr))
        hits = []
        for a, b in segs:
            dur = (b - a + 1) * FRAME
            if not (MIN_LEN <= dur <= MAX_LEN):
                continue
            rel = db(rms[a:b + 1].max()) - db(full)
            if rel < MIN_DB_REL:
                continue          # 相对全篇太弱 -> 底噪，不是真的音
            hits.append((a * FRAME, (b + 1) * FRAME, dur,
                         float(up[a:b + 1].max()), rel))
        if not hits:
            print("  ✅ 无「说完马上冒出来的短突起」")
            continue
        print("  候选 %d 处（已排除底噪级）：" % len(hits))
        for t0, t1, dur, upv, rel in hits:
            print("    %.2fs ~ %.2fs  时长 %.2fs  高出局部基线 %+.1f dB  "
                  "相对全篇 %+.1f dB" % (t0, t1, dur, upv, rel))
    return 0


if __name__ == "__main__":
    sys.exit(main())
