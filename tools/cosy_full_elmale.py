# -*- coding: utf-8 -*-
"""决定性实验：裸推理这条链上，把参考从 10s 的 prompt_yohan_1a6c20a6.wav
换成现役 15s 的 EL_male.wav，其他一切不变。

为什么跑它（2026-10-13）：
  同一案例 1a6c20a60481、同一文案、同一参照：
    裸推理（10s prompt_yohan_1a6c20a6.wav）  0.9532   445.7s
    线上成品（15s EL_male.wav）              0.8444   490.1s
  差 0.109。逐项归因已证明后处理（atempo/mp3/slowdyn/EQ）合计只吃 0.0013。
  剩下的嫌疑只有两个：①参考音频不同 ②线上主服务的长文案分块/处理不同。

  本实验只换参考音频，走**同一条裸推理链**：
    结果 ≈0.95 → 参考无罪，0.109 来自线上主服务的长文案处理；
    结果 ≈0.85 → 参考是主因，换掉 EL_male 的 15s 切片就是白捡的 +0.11。

输出：/mnt/tts/projects/gpt-sovits/out/cosy_full_m_elmale.wav
"""
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
OUT = "/mnt/tts/projects/gpt-sovits/out"
DST = os.path.join(OUT, "cosy_full_m_elmale.wav")


def main():
    txt = open(os.path.join(OUT, "full_1a6c20a60481.txt"),
               encoding="utf-8").read().strip()
    # 现役生产参考：voices.json 里 EL_male 的 15s wav + 它的逐字稿
    import json
    v = json.load(open("/mnt/tts/data/voices/voices.json"))["EL_male"]
    pwav, ptx = v["path"], (v.get("ref_text") or "")
    print(f"文案 {len(txt)} 字  参考 {pwav}  ref_text {len(ptx)} 字", flush=True)

    from app.cosy_client import synth
    audio, sr, meta = synth(text=txt, ref_wav=pwav, ref_text=ptx,
                            instruct="neutral", speed=1.0, seed=42)
    if audio is None:
        print("❌ 生成失败:", meta)
        return 1
    sf.write(DST, np.asarray(audio, dtype="float32"), sr, subtype="PCM_16")
    print(f"-> {DST}  {len(audio)/sr:.1f}s @{sr}Hz", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
