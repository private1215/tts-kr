#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""块长对照实验（VOX_CHUNK_CHARS：600 / 300 / 150）—— 男声 VoxCPM2 直连

## 依据
`steps_ab.py` 结果（600 字一块，10s 窗，campplus 对 EL_male）：

| 档 | 音色 前/中/后 | 极差 | 末−首 |
|---|---|---|---|
| s10(现役) | 0.867 / 0.789 / 0.735 | 0.147 | −0.130 |
| s20 | 0.841 / 0.762 / 0.730 | 0.150 | −0.150 |
| s30 | 0.824 / 0.689 / 0.648 | **0.208** | **−0.208** |

→ ❌ **提 STEPS 无效且更差**（别再提步数，10 步就是最优）
→ ✅ **块内音色衰减坐实**：一块 600 字里，从头到尾音色掉 ~0.13，
   块结束后重新注入 prompt 又回到 0.87 → 长篇就是一串"锯齿"，
   这正是窗间极差 0.41（我们）vs 0.042（EL）的来源。

假设：**块越短 → 衰减幅度越小 → 整体音色更稳、均值更高**。
（旧结论"分块背锅已证伪 0.9947"测的是**块间接缝**，
  没测块内衰减，两者不冲突。）

## 用法
  /mnt/tts/projects/voxcpm/venv/bin/python chunk_ab.py
评价：
  DRIFT_WIN=10 /mnt/tts/projects/qwen3-tts/venv/bin/python /mnt/tts/diag/drift_win.py <wav>
🔴 L1 快筛：只看"块长有没有用"，上线前必须长篇回归 + probe_g。
"""
import os
import re
import sys
import time

import numpy as np
import soundfile as sf

BASE = "/mnt/tts/projects/voxcpm"
sys.path.insert(0, BASE)

OUT = "/mnt/tts/diag/_chunk"
os.makedirs(OUT, exist_ok=True)

REF_WAV = "/mnt/tts/projects/gpt-sovits/out/prompt_yohan_1a6c20a6.wav"
REF_TXT = "/mnt/tts/projects/gpt-sovits/out/prompt_yohan_1a6c20a6.txt"
TEXT_FILE = "/mnt/tts/data/cases/6cdd25bc75f2/seg0.txt"
N_CHARS = 1200        # 固定总长（≈130s），只改块长
MODEL_DIR = os.path.join(BASE, "models/VoxCPM2")
STEPS, CFG = 10, 2.0  # 现役值

VARIANTS = [("c600", 600), ("c300", 300), ("c150", 150)]


def split_sentences(text, limit):
    """与 vox_server.py 完全一致（保证结论可迁移到线上）"""
    parts = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) + 1 > limit and cur:
            chunks.append(cur.strip())
            cur = p
        else:
            cur = (cur + " " + p).strip()
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def main():
    if not os.path.exists(REF_WAV):
        raise SystemExit("缺参考音频 %s" % REF_WAV)
    rtext = open(REF_TXT, encoding="utf-8").read().strip()
    text = open(TEXT_FILE, encoding="utf-8").read().strip()[:N_CHARS]

    from voxcpm.core import VoxCPM
    m = VoxCPM(voxcpm_model_path=MODEL_DIR, enable_denoiser=False)
    sr = int(getattr(m.tts_model, "sample_rate", 16000))
    print("模型已加载 sr=%d  总文本 %d 字\n" % (sr, len(text)), flush=True)

    for tag, limit in VARIANTS:
        chunks = split_sentences(text, limit)
        wav = os.path.join(OUT, "%s.wav" % tag)
        t0 = time.time()
        pieces = []
        for i, c in enumerate(chunks):
            try:
                a = m.generate(text=c, reference_wav_path=REF_WAV,
                               prompt_wav_path=REF_WAV, prompt_text=rtext,
                               cfg_value=CFG, inference_timesteps=STEPS,
                               normalize=True)
            except Exception as ex:
                print("%s 第%d块失败: %r" % (tag, i, ex), flush=True)
                break
            arr = np.asarray(a)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            pieces.append(arr)
        if len(pieces) != len(chunks):
            continue
        out = np.concatenate(pieces)
        sf.write(wav, out.astype(np.float32), sr, subtype="PCM_16")
        print("%-5s 块长=%-4d 块数=%d -> %.1fs  用时 %.0fs  %s"
              % (tag, limit, len(chunks), len(out) / float(sr),
                 time.time() - t0, wav), flush=True)
    print("\nCHUNK_DONE")


if __name__ == "__main__":
    main()
