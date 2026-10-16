#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""repro47.py —— 手工复现 fix_chain 的**必经路径**，逐步定位 click 从哪一步出现。

fix_chain 在 `--no-clean --no-pitch --eq anull` 下（step1 = 0.54%，归一后）
理论上只剩这些必经步骤，本脚本逐个复现并测量：

    A  load16      ffmpeg -ar 16000 -ac 1
    B  sf.write/sf.read 往返（numpy 进出一次）
    C  ffmpeg -af anull
    D  ffmpeg -af volume=1.4dB
    E  ffmpeg -af anull,volume=1.4dB（等价于 step1 的主增益步）

每步都做**峰值归一**后再判 click，避免踩绝对阈值的坑。
"""
import os
import subprocess

import numpy as np
import soundfile as sf

SR = 16000
OUT = "/mnt/tts/diag/repro47"
RAW = "/mnt/tts/diag/raw47/step0_raw.wav"


def load(p):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-ar", str(SR),
                        "-ac", "1", "-f", "f32le", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.float32)


def norm(x, peak=0.70):
    m = np.abs(x).max()
    return x * (peak / m) if m > 1e-9 else x


def click(x, thr=0.30):
    n = int(SR * 0.02)
    d = np.abs(np.diff(x))
    m = (len(d) // n) * n
    dmax = d[:m].reshape(-1, n).max(axis=1)
    return int((dmax >= thr).sum()), len(dmax)


def codec(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "stream=codec_name,sample_rate,bits_per_raw_sample",
                        "-of", "default=nw=1", p],
                       capture_output=True, text=True)
    return " ".join(r.stdout.split())


def report(tag, p):
    k, tot = click(norm(load(p)))
    print(f"{tag:<26}{k:>7}{k/tot*100:>8.2f}%   [{codec(p)}]")


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"{'步骤':<26}{'click':>7}{'占比':>9}")
    print("-" * 76)

    report("参照 引擎裸(step0)", RAW)

    # A: load16
    a = f"{OUT}/A.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", RAW,
                    "-ar", "16000", "-ac", "1", a], check=True)
    report("A  load16 (-ar 16000)", a)

    # B: sf.write / sf.read 往返
    y, sr = sf.read(a, dtype="float32")
    b = f"{OUT}/B.wav"
    sf.write(b, y, sr)
    report("B  sf.write/sf.read 往返", b)

    # C: ffmpeg anull
    c = f"{OUT}/C.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", b,
                    "-af", "anull", c], check=True)
    report("C  ffmpeg -af anull", c)

    # D: ffmpeg volume
    d = f"{OUT}/D.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", c,
                    "-af", "volume=1.4dB", d], check=True)
    report("D  + volume=1.4dB", d)

    # E: anull,volume 一步
    e = f"{OUT}/E.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", b,
                    "-af", "anull,volume=1.4dB", e], check=True)
    report("E  anull,volume 一步", e)

    # 对照：fix_chain 产物
    report("fix_chain step1", "/mnt/tts/diag/raw47/step1.wav")

    # 关键对比：A 与 B 是否有实质差异
    xa, xb = load(a), load(b)
    n = min(len(xa), len(xb))
    xa, xb = xa[:n], xb[:n]
    g = (xa @ xb) / (xa @ xa)
    res = xb - g * xa
    snr = 10 * np.log10((xb @ xb) / max(res @ res, 1e-30))
    print(f"\nA->B 残差 SNR = {snr:.2f} dB（numpy 往返有没有损伤）")


if __name__ == "__main__":
    main()
