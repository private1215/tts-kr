# -*- coding: utf-8 -*-
"""q1_template_click.py —— Q1：模版自身比成品脏，到底是什么情况。

要回答三件事：
1. 模版的 7 处 click 在**原生 24kHz**（不重采样）下是否依然存在
   —— 排除 24k→16k 重采样本身制造假 click 的可能；
2. 每处 click 的精确时间与幅度，特别是 14.556~14.561s 那一簇（文件尾）是什么；
3. 同一口径下对比：模版（EL 原始 15s） vs EL 长文本标杆 merged.wav
   vs 我们的成品 202610190902 —— 说明「模版脏」不代表「我们链路脏」。
"""
import os
import subprocess

import numpy as np
import soundfile as sf

DIAG = "/mnt/tts/diag/v0902"
TPL = "/mnt/tts/data/voices/EL_female.wav"
EL_BENCH = "/mnt/tts/diag/v0902/hm_merged.wav.wav"     # EL 女声长文本标杆 16k
OURS = "/mnt/tts/diag/v0902/202610190902-1.mp3.16k.wav"  # 我们成品 16k


def find_clicks(x, sr, th=0.30, target_pk=0.70):
    """现役口径：峰值归一到 0.70，|Δ|>=0.30。返回 (t, |Δ|) 列表（合并相邻帧）。"""
    pk = float(np.max(np.abs(x)))
    if pk <= 0:
        return []
    y = x * (target_pk / pk)
    d = np.abs(np.diff(y))
    idx = np.where(d >= th)[0]
    out = []
    for i in idx:
        if out and (i - out[-1][0] * sr) <= 2:   # 同一处跳变的相邻样本
            if d[i] > out[-1][1]:
                out[-1] = (i / sr, float(d[i]))
            continue
        out.append((i / sr, float(d[i])))
    return out


def tail_region(x, sr, t0, t1):
    """打印文件尾区域的 10ms 能量包络，看那一簇 click 前后是什么形态。"""
    i0, i1 = int(t0 * sr), min(int(t1 * sr), len(x))
    seg = x[i0:i1]
    hop = int(0.01 * sr)
    print("   %-9s %s" % ("t(s)", "10ms 包络 dBFS"))
    for i in range(0, len(seg) - hop, hop):
        v = seg[i:i + hop]
        db = 20 * np.log10(np.sqrt(np.mean(v ** 2)) + 1e-12)
        bar = "#" * max(0, int((db + 60) / 1.5))
        print("   %-9.3f %6.1f %s" % (t0 + i / sr, db, bar))


def report(path, label, native=False):
    if native:
        x, sr = sf.read(path, dtype="float32")
    else:
        w = "/tmp/q1_%s.wav" % os.path.basename(path)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                        "-ar", "16000", "-ac", "1", w], check=True)
        x, sr = sf.read(w, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    dur = len(x) / sr
    cl = find_clicks(x, sr)
    print("\n===== %s =====" % label)
    print("时长 %.3f s  采样率 %d Hz  click %d 处（%.3f/分钟）"
          % (dur, sr, len(cl), len(cl) / (dur / 60)))
    for t, m in cl:
        print("   t=%8.3f  |Δ|=%.3f" % (t, m))
    return x, sr, cl


def main():
    print("########## 1. 模版：原生 24kHz vs 重采样 16k ##########")
    x24, sr24, cl24 = report(TPL, "模版 EL_female.wav 原生 24kHz", native=True)
    x16, sr16, cl16 = report(TPL, "模版 EL_female.wav 重采样到 16k")

    same = len(cl24) == len(cl16)
    print("\n→ 24kHz %d 处 / 16kHz %d 处，%s"
          % (len(cl24), len(cl16),
             "数量一致（click 不是重采样伪影）" if same else "数量不一致（重采样有影响）"))
    for t, m in cl24:
        near = [ (t2,m2) for t2,m2 in cl16 if abs(t2-t) < 0.01 ]
        print("   24k t=%.3f Δ=%.3f  ↔ 16k %s"
              % (t, m, ("t=%.3f Δ=%.3f" % near[0]) if near else "无对应"))

    print("\n########## 2. 文件尾 14.5~15.1s 区域细看（24kHz） ##########")
    tail_region(x24, sr24, 14.40, 15.08)

    print("\n########## 3. 文件头 0~0.6s 区域细看（24kHz） ##########")
    tail_region(x24, sr24, 0.0, 0.6)

    print("\n########## 4. 同口径横向对比（全部 16k） ##########")
    for p, lab in ((EL_BENCH, "EL 女声长文本标杆 merged.wav"),
                   (OURS, "我们成品 202610190902.mp3")):
        if not os.path.exists(p):
            print("  [跳过] %s 不存在" % lab)
            continue
        _, _, cl = report(p, lab)
        print("   → %s：%d 处" % (lab, len(cl)))


if __name__ == "__main__":
    main()
