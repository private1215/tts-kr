# -*- coding: utf-8 -*-
"""CosyVoice3 侧：**案例库完整长文案**对照生成。

🔴 口径必须与 GPT-SoVITS 侧**完全一致**，否则三引擎分数不可比：
  · 文案：同一份 full_<cid>.txt（案例全部片段拼起来的完整稿）
  · 参照：同一段 prompt_<spk>_<cid>.wav/txt（取自案例开头，不入训练区）
  · 后处理：都走线上同一条 pipeline（响度闭环 / peakfit / PBA）
不同的只有「哪个引擎念」。

为什么参照要跟着文案走：案例库实测只有 2 个核心声音（SOLA 女 22 条 /
yohan 男 13 条），但每次上传的**切片质量不同**。拿 A 案例的切片去念
B 案例的稿，音色相似度会被切片好坏污染，而不是引擎差异。

用法：
    CUDA_VISIBLE_DEVICES=... python -u tools/cosy_full.py
"""
import os
import sys

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")

import numpy as np                                     # noqa: E402
import soundfile as sf                                 # noqa: E402

OUT = "/mnt/tts/projects/gpt-sovits/out"

# (输出名, 文案, 参照wav, 参照txt, instruct)
JOBS = [
    ("cosy_full_f",
     os.path.join(OUT, "full_00b6a45ad6aa.txt"),
     os.path.join(OUT, "prompt_sola_00b6a45a.wav"),
     os.path.join(OUT, "prompt_sola_00b6a45a.txt"),
     "neutral"),
    ("cosy_full_m",
     os.path.join(OUT, "full_1a6c20a60481.txt"),
     os.path.join(OUT, "prompt_yohan_1a6c20a6.wav"),
     os.path.join(OUT, "prompt_yohan_1a6c20a6.txt"),
     "neutral"),
]


def main():
    from app.cosy_client import synth
    for name, tpath, pwav, ptxt, ins in JOBS:
        dst = os.path.join(OUT, name + ".wav")
        if os.path.exists(dst):
            print("  跳过(已存在)", name, flush=True)
            continue
        txt = open(tpath, encoding="utf-8").read().strip()
        ptx = open(ptxt, encoding="utf-8").read().strip()
        print("=== %s  字数=%d  参照=%s ===" % (name, len(txt), ptx[:30]),
              flush=True)
        audio, sr, meta = synth(text=txt, ref_wav=pwav, ref_text=ptx,
                                instruct=ins, speed=1.0, seed=42)
        if audio is None:
            print("   ❌ 失败:", meta.get("error"), flush=True)
            continue
        audio = np.asarray(audio, dtype="float32")
        sf.write(dst, audio, sr, subtype="PCM_16")
        print("   -> %s  %.1fs @%dHz" % (dst, len(audio) / sr, sr),
              flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
