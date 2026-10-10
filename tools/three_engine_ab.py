# -*- coding: utf-8 -*-
"""三引擎对照：Qwen3-TTS / CosyVoice3 / GPT-SoVITS 同一样本、同一文案。

为什么必须同一 Worker：
- 参照音频统一用 **out/prompt_10s.wav**（案例前 60s 里的 5~15s，测试区，GS 没训练过）
  GS 和 Cosy 都是零样本/微调克隆，**参照给得多的一方天然占便宜**，不统一就说明不了问题。
- Qwen 侧走 EL_female（线上默认克隆音色，本身也是从这个案例建的），保持"员工今天
  听到的就是它"。

统一评判口径交给 tools/three_engine_score.py：说话人相似度 / CER / 韵律贴合度 / 时长。
单项好看不算数，必须四个一起看 —— 本项目已经因为只看单项翻过两次车。

用法：
    python tools/three_engine_ab.py
"""
import os
import sys

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
os.chdir("/mnt/tts/projects/qwen3-tts")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from app.cosy_client import synth  # noqa: E402

TEXT = open("/mnt/tts/diag/hh_short.txt", encoding="utf-8").read().strip()
REF_WAV = "/mnt/tts/projects/gpt-sovits/out/prompt_10s.wav"
REF_TXT = open("/mnt/tts/projects/gpt-sovits/out/prompt_10s.txt", encoding="utf-8").read().strip()
OUT = "/mnt/tts/projects/gpt-sovits/out"

INSTRUCTS = [
    ("neutral", "cosy_zs.mp3"),
    ("excited", "cosy_exc.mp3"),
]


def main():
    print("参照:", REF_WAV, "|", REF_TXT, flush=True)
    print("文案 %d 字" % len(TEXT), flush=True)
    for ins, name in INSTRUCTS:
        print("=== CosyVoice3 instruct=%s ===" % ins, flush=True)
        audio, sr, meta = synth(
            text=TEXT, ref_wav=REF_WAV, ref_text=REF_TXT,
            instruct=ins, speed=1.0, seed=42,
        )
        if audio is None:
            print("   失败:", meta.get("error"), flush=True)
            continue
        audio = np.asarray(audio, dtype=np.float32)
        p = os.path.join(OUT, name.replace(".mp3", ".wav"))
        sf.write(p, audio, sr, subtype="PCM_16")
        print("   -> %s  %.1fs @%dHz" % (p, len(audio) / sr, sr), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
