#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只算 campplus 音色余弦：变调前后 vs EL 参考。

问题：VoxCPM2 男声 F0 比 EL 低 15%（167.8 vs ~198），用 rubberband 升 1.18 补回。
本脚本验证变调有没有把音色搞坏（campplus 是否掉）。
"""
import os
import sys

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/tools")

import numpy as np  # noqa: E402
import engine_score_all as ES  # noqa: E402

REF = "/mnt/tts/data/voices/EL_male.wav"
# 另外拿两条 EL 案例当第二参照，避免只用 15s 参考片
REFS = [("EL_male参考", REF),
        ("案例-男声", "/mnt/tts/data/cases/1a6c20a60481/merged.wav"),
        ("案例8197", "/mnt/tts/data/cases/8197d9ac8a35/merged.wav")]
CANDS = [
    ("cf9原样(未变调)", "/mnt/tts/diag/cf9/vox_male_fix.wav"),
    ("cf9变调1.18", "/mnt/tts/diag/cf9/vox_male_pitch.wav"),
    ("员工听到的cosy3", "/mnt/tts/output/안녕하세요, 필옵틱스 주주 여러분, 반갑습니/안녕하세요, 필옵틱스 주주 여러분, 반갑습니-1.mp3"),
]


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


embs = {}
for n, p in REFS + CANDS:
    embs[n] = ES.timbre(p)
    print("embedded", n, flush=True)

print()
print("候选 \\ 参照\t" + "\t".join(n for n, _ in REFS))
for cn, _ in CANDS:
    row = [cn]
    for rn, _ in REFS:
        row.append("%.4f" % cos(embs[cn], embs[rn]))
    print("\t".join(row))
