#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""哪个声音更像 EL 标杆？（campplus 余弦相似度，多窗平均）

第30轮员工说「声色完全变了」——因为我们把男声从 omnivoice（其实是女声）
修回了 voxcpm（真男声）。这里给出客观依据：两个引擎各自离 EL 男标杆多远。

用法（必须用 cosy 的 venv，只有它有 kaldi fbank + onnxruntime）：
    /mnt/tts/projects/cosy/venv/bin/python tools/who_closer.py
"""
import os
import subprocess
import sys

import numpy as np

SR = 16000
WIN = 10.0

PAIRS = [
    ("EL 男标杆 6cdd", "/mnt/tts/data/cases/6cdd25bc75f2/merged.wav"),
    ("新男声 voxcpm(r30)", "/mnt/tts/output/longprobe-r30vox-6cdd-3/longprobe-r30vox-6cdd-1.mp3"),
    ("老男声 voxcpm(无declick)", "/mnt/tts/output/longprobe-6cdd/longprobe-6cdd-1.mp3"),
    ("旧『男声』=omnivoice 女声", "/mnt/tts/output/longprobe-omni-1block/longprobe-omni-1block-1.mp3"),
]


def load16(p):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", p,
                          "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def main():
    sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")
    from campplus_extract import EmbeddingExtractor  # 若不存在则走下面分支
    ex = EmbeddingExtractor()
    emb = {}
    for name, p in PAIRS:
        if not os.path.exists(p):
            print("缺:", p)
            continue
        y = load16(p)
        n = int(SR * WIN)
        nw = max(1, len(y) // n)
        vs = []
        for i in range(nw):
            seg = y[i * n:(i + 1) * n]
            if len(seg) < SR:
                continue
            e = ex.inference(seg[None, :])
            e = np.asarray(e).reshape(-1)
            vs.append(e / (np.linalg.norm(e) + 1e-9))
        v = np.mean(vs, axis=0)
        emb[name] = v / (np.linalg.norm(v) + 1e-9)
        print("  %-26s %d 窗" % (name, len(vs)))
    ref = emb.get("EL 男标杆 6cdd")
    if ref is None:
        return 1
    print("\n与 EL 男标杆的余弦相似度：")
    for name in emb:
        print("  %-26s %.4f" % (name, float(emb[name] @ ref)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
