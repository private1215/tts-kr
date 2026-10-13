# -*- coding: utf-8 -*-
"""VoxCPM2 冒烟测试：确认模型能加载、韩语参考克隆能出声。

用法（服务器）：
    CUDA_VISIBLE_DEVICES=GPU-b67eee2c-... python -u tools/voxcpm_smoke.py
"""
import os
import sys
import time

import numpy as np
import soundfile as sf

BASE = "/mnt/tts/projects/voxcpm"
MODEL = os.path.join(BASE, "models/VoxCPM2")
GSOUT = "/mnt/tts/projects/gpt-sovits/out"
OUT = os.path.join(BASE, "smoke")
os.makedirs(OUT, exist_ok=True)

TEXT = "안녕하세요. 오늘도 변동성이 큰 장세였습니다. 꾸준함이 결국 승부를 가릅니다."

JOBS = [
    # (输出名, 参考模式: ref-only / prompt+ref, 参考wav, 参考文本)
    ("smoke_yohan_ref", "ref", os.path.join(GSOUT, "prompt_yohan_1a6c20a6.wav"), None),
    ("smoke_yohan_prompt", "prompt",
     os.path.join(GSOUT, "prompt_yohan_1a6c20a6.wav"), None),
]


def main():
    from voxcpm.core import VoxCPM
    t0 = time.time()
    model = VoxCPM(voxcpm_model_path=MODEL, enable_denoiser=False,
                   optimize=True, device="cuda")
    print("加载用时 %.1fs" % (time.time() - t0), flush=True)

    for name, mode, rw, _ in JOBS:
        t1 = time.time()
        kw = dict(text=TEXT, reference_wav_path=rw,
                  cfg_value=2.0, inference_timesteps=10, normalize=True)
        if mode == "prompt":
            kw["prompt_wav_path"] = rw
            kw["prompt_text"] = open(os.path.join(
                GSOUT, "prompt_yohan_1a6c20a6.txt"), encoding="utf-8").read().strip()
        try:
            audio = model.generate(**kw)
            sr = int(model.tts_model.sample_rate)
        except Exception as e:
            print("%s 失败: %r" % (name, e), flush=True)
            continue
        if audio is None:
            print("%s 返回空" % name, flush=True)
            continue
        audio = np.asarray(audio)
        dst = os.path.join(OUT, name + ".wav")
        sf.write(dst, audio, sr, subtype="PCM_16")
        print("%s -> %.1fs 用时 %.1fs (%s sr=%d)"
              % (dst, len(audio) / float(sr), time.time() - t1, mode, sr),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
