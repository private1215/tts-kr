#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""number_fix.py — 数字段语速对齐 EL（第七轮：电话号码念得含混）。
实测：我们的数字区 214.05-215.85s（1.80s，约154ms/位），EL 数字区
212.63-214.63s（2.00s，约182ms/位）→ 我们的快 15%，且后半段间隔挤到
0.13-0.15s（EL 均匀 0.23-0.27s）。
修法：数字段整体 atempo 拉伸 1.18×（= 182/154），嵌回时用 10ms 交叉淡化
（切点都落在号码前后的静音里，无语音损伤）。
用法: number_fix.py <in.wav> <out.wav> [起] [止] [倍率]
"""
import sys, os, subprocess, tempfile
import numpy as np
import soundfile as sf

def main():
    inp = sys.argv[1]
    outp = sys.argv[2]
    a = float(sys.argv[3]) if len(sys.argv) > 3 else 214.05
    b = float(sys.argv[4]) if len(sys.argv) > 4 else 215.85
    ratio = float(sys.argv[5]) if len(sys.argv) > 5 else 1.18

    y, sr = sf.read(inp)
    mono = y.ndim == 1
    wav = y if mono else y.mean(axis=1)
    i0, i1 = int(a * sr), int(b * sr)
    seg = wav[i0:i1]
    print(f"数字段 [{a}~{b}] {len(seg)/sr:.2f}s → 拉伸 {ratio}x")

    td = tempfile.mkdtemp()
    t1 = os.path.join(td, "seg.wav")
    t2 = os.path.join(td, "seg_s.wav")
    sf.write(t1, seg, sr)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", t1,
                    "-af", f"atempo={1.0/ratio:.5f}", t2], check=True)
    seg2, _ = sf.read(t2, dtype="float32")
    print(f"拉伸后 {len(seg2)/sr:.2f}s（+{(len(seg2)-len(seg))/sr:.2f}s）")

    w = max(1, int(0.01 * sr))
    ramp = np.linspace(0, 1, w)
    left, right = wav[:i0], wav[i1:]
    # 左端：原尾部 w 样本 ↔ 新头部 w 样本 交叉淡化
    tail = left[-w:].copy()
    head = seg2[:w].copy()
    # 右端：新尾部 ↔ 原头部
    tail2 = seg2[-w:].copy()
    head2 = right[:w].copy()
    new = np.concatenate([
        left[:-w],
        tail * (1 - ramp) + head * ramp,
        seg2[w:-w],
        tail2 * ramp[::-1] + head2 * (1 - ramp[::-1]),
        right[w:],
    ])
    print(f"全长 {len(wav)/sr:.2f}s → {len(new)/sr:.2f}s")
    if mono:
        sf.write(outp, new, sr)
    else:
        # 多声道防御：对每声道做同样替换（本流程单声道）
        chans = []
        for c in range(y.shape[1]):
            v = y[:, c]
            chans.append(np.concatenate([v[:i0 - w],
                                         v[i0 - w:i0] * (1 - ramp) + seg2[:w] * ramp,
                                         seg2[w:-w],
                                         seg2[-w:] * ramp[::-1] + v[i1:i1 + w] * (1 - ramp[::-1]),
                                         v[i1 + w:]]))
        sf.write(outp, np.stack(chans, axis=1), sr)
    print("saved:", outp)

main()
