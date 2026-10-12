# -*- coding: utf-8 -*-
"""CosyVoice3 侧：**案例库完整长文案**对照生成。

⛔⛔ 2026-10-12 警告：**不要再用这个脚本做跨引擎对比**
------------------------------------------------------
它产出的是 **CLI 裸推理音频**（`cosy_client.synth` 直接 `sf.write`），
**没有走后处理链**（PBA 停顿对齐 / 慢速修复 / EQ / PROSO 韵律移植 /
呼吸注入 / 响度闭环 / peakfit）。而线上 Qwen 的音频是走完整链的成品。
拿它和线上成品比韵律，等于让 Cosy 裸奔上场 —— 上一轮长文案对照就这么错过一次。

⚠️ 旧版注释写着"后处理：都走线上同一条 pipeline"，**那是错的**，
   注释和实现不一致正是这次犯错的根源。已改正。

✅ 跨引擎对比请用 `tools/engine_full_cmp.py`：所有引擎都走 `/api/jobs` 提交，
   由主服务跑同一条流水线，只有"谁来念"不同。

✅ 这个脚本现在只剩两个正当用途：
   1. **只比 Cosy 内部**（不同 seed / instruct / 语速），口径一致；
   2. 快速验证"服务通不通、能不能出片"。

🔴 参照要跟着文案走
------------------
案例库实测只有 2 个核心声音（SOLA 女 22 条 / yohan 男 13 条），
但每次上传的**切片质量不同**。拿 A 案例的切片去念 B 案例的稿，
音色相似度会被切片好坏污染，而不是引擎差异。

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
