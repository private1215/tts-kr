# -*- coding: utf-8 -*-
"""OmniVoice 冒烟测试：韩语参考克隆能否出声。"""
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

BASE = "/mnt/tts/projects/omnivoice"
MODEL = os.path.join(BASE, "models/OmniVoice")
GSOUT = "/mnt/tts/projects/gpt-sovits/out"
OUT = os.path.join(BASE, "smoke")
os.makedirs(OUT, exist_ok=True)

TEXT = "안녕하세요. 오늘도 변동성이 큰 장세였습니다. 꾸준함이 결국 승부를 가릅니다."


def main():
    from omnivoice import OmniVoice
    t0 = time.time()
    model = OmniVoice.from_pretrained(MODEL, device_map="cuda",
                                      dtype=torch.float16)
    print("加载 %.1fs" % (time.time() - t0), flush=True)
    rw = os.path.join(GSOUT, "prompt_yohan_1a6c20a6.wav")
    rt = open(os.path.join(GSOUT, "prompt_yohan_1a6c20a6.txt"),
              encoding="utf-8").read().strip()
    t1 = time.time()
    try:
        audio = model.generate(text=TEXT, ref_audio=rw, ref_text=rt)
    except Exception as e:
        print("带ref_text失败: %r，尝试省略ref_text" % (e,), flush=True)
        audio = model.generate(text=TEXT, ref_audio=rw)
    print("用时 %.1fs 返回 %d 段" % (time.time() - t1, len(audio)), flush=True)
    dst = os.path.join(OUT, "smoke_yohan.wav")
    sf.write(dst, np.asarray(audio[0]), 24000, subtype="PCM_16")
    print("-> %s %.1fs" % (dst, len(np.asarray(audio[0])) / 24000.0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
