#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEPS / CFG 对照实验（男声 VoxCPM2，直连模型，不动线上服务 :8400）

## 为什么要做这个
第35轮员工反馈（3 人，控制题全对）：
- 我们的 4 段评级均分 2.67 / 2.67 / 2.0 / **3.00（尾部最差，3/3 全票"明显电子"）**
- EL 隐藏对照 1.33（像真人 / 有点机器感 / 像真人）→ 标尺没失灵

客观验证（drift_win.py，20s 窗，campplus 对 EL_male）：
- 我们 h5：每窗音色 **0.451 ~ 0.863，极差 0.412**，尾部 440s 窗只有 0.548
- 我们 r35（h5 之前）：极差 0.402 → **不是 h5 引入的**
- **EL 案例：0.911 ~ 0.954，极差 0.042** ← 我们波动是 EL 的 10 倍
- F0 / 发声占比 / 频段都接近 → 唯一拉开的就是"音色稳不稳"

已排除：vox_server 每 600 字就重新注入 prompt（`prompt_wav_path=rw`），
不是"参考忘掉"，是**块内生成噪声**。现役 `VOX_STEPS=10` → 头号嫌疑。

## 用法
只做生成（voxcpm venv）：
  /mnt/tts/projects/voxcpm/venv/bin/python steps_ab.py
评价另跑（qwen3-tts venv，避免 kaldi 依赖冲突）：
  DRIFT_WIN=10 /mnt/tts/projects/qwen3-tts/venv/bin/python /mnt/tts/diag/drift_win.py <wav>
🔴 L1 快筛用途：只看"改动有没有生效"，验收必须长篇。
"""
import os
import sys
import time

import numpy as np
import soundfile as sf

BASE = "/mnt/tts/projects/voxcpm"
sys.path.insert(0, BASE)

OUT = "/mnt/tts/diag/_steps"
os.makedirs(OUT, exist_ok=True)

REF_WAV = "/mnt/tts/projects/gpt-sovits/out/prompt_yohan_1a6c20a6.wav"
REF_TXT = "/mnt/tts/projects/gpt-sovits/out/prompt_yohan_1a6c20a6.txt"
TEXT_FILE = "/mnt/tts/data/cases/6cdd25bc75f2/seg0.txt"
N_CHARS = 600          # 与线上 VOX_CHUNK_CHARS 一致（≈60s）
MODEL_DIR = os.path.join(BASE, "models/VoxCPM2")

VARIANTS = [
    ("s10", 10, 2.0),   # 现役
    ("s20", 20, 2.0),
    ("s30", 30, 2.0),
]


def main():
    if not os.path.exists(REF_WAV):
        raise SystemExit("缺参考音频 %s" % REF_WAV)
    rtext = open(REF_TXT, encoding="utf-8").read().strip()
    text = open(TEXT_FILE, encoding="utf-8").read().strip()[:N_CHARS]
    print("参考: %s（%d 字）\n合成: %d 字\n" % (os.path.basename(REF_WAV), len(rtext), len(text)))

    from voxcpm.core import VoxCPM
    m = VoxCPM(voxcpm_model_path=MODEL_DIR, enable_denoiser=False)
    # 🔴 真实采样率在 tts_model 上（vox_server:_sr = _model.tts_model.sample_rate），
    #    取 m.sample_rate 会拿到默认值 16000 → 时长被放大 ~2.7 倍（踩过）
    sr = int(getattr(m.tts_model, "sample_rate", 16000))
    print("模型已加载 sr=%d\n" % sr, flush=True)

    for tag, steps, cfg in VARIANTS:
        wav = os.path.join(OUT, "%s.wav" % tag)
        t0 = time.time()
        try:
            a = m.generate(text=text, reference_wav_path=REF_WAV,
                           prompt_wav_path=REF_WAV, prompt_text=rtext,
                           cfg_value=cfg, inference_timesteps=steps,
                           normalize=True)
        except Exception as ex:
            print("%s 生成失败: %r" % (tag, ex), flush=True)
            continue
        arr = np.asarray(a)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        sf.write(wav, arr.astype(np.float32), sr, subtype="PCM_16")
        print("%-4s steps=%-3d cfg=%.1f -> %.1fs  用时 %.0fs  %s"
              % (tag, steps, cfg, len(arr) / float(sr), time.time() - t0, wav),
              flush=True)
    print("\nSTEPS_DONE")


if __name__ == "__main__":
    main()
