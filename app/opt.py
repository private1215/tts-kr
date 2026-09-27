# -*- coding: utf-8 -*-
"""网格测试：分段长度 x 批量大小 对 RTF / 显存 的影响。

假设：attention 随序列长度平方增长，短片段 + 大批量的吞吐显著优于长片段 + 小批量。
"""
import os
import sys
import time
import json

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_engine import TTSEngine
from pipeline import build_segments

MODEL_DIR = "/mnt/tts/models"
MODEL_DIRS = {
    "custom_voice": os.path.join(MODEL_DIR, "Qwen3-TTS-12Hz-1.7B-CustomVoice"),
    "base": os.path.join(MODEL_DIR, "Qwen3-TTS-12Hz-1.7B-Base"),
}

PARA = [
    "요즘 인공지능 기술은 정말 빠르게 발전하고 있습니다. 특히 음성 합성 분야에서는 사람의 목소리와 거의 구별하기 어려운 수준까지 도달했습니다.",
    "하지만 아직도 긴 문장을 읽을 때는 어딘가 어색한 느낌이 남아 있습니다. 그 이유는 기계가 문장의 감정과 호흡을 균일하게 처리하기 때문입니다.",
    "사람은 문장을 읽을 때 의미 단위마다 속도를 바꾸고, 중요한 부분에서 잠시 멈춥니다. 이런 미세한 변화가 모여서 자연스러운 말하기가 완성됩니다.",
    "그래서 우리는 긴 원고를 한 번에 읽히는 대신, 의미 단위로 나누어 각각의 호흡을 부여합니다. 이렇게 하면 전체적으로 훨씬 자연스럽습니다.",
    "한국어는 특히 문장 끝의 억양이 중요합니다. 같은 문장이라도 끝을 올리면 질문이 되고, 내리면 단정적인 표현이 됩니다.",
    "또한 한국어에는 한자어와 외래어가 많습니다. 이런 단어들을 어떻게 발음하느냐에 따라서도 전체적인 인상이 달라집니다.",
    "숫자나 단위가 들어간 문장도 주의가 필요합니다. 쉰 퍼센트라고 읽을지 오십 퍼센트라고 읽을지는 문맥에 따라 다릅니다.",
    "결국 좋은 음성을 만들기 위해서는 모델의 성능도 중요하지만, 원고를 어떻게 준비하느냐도 매우 중요합니다.",
]

# 拼足足够长的文本，保证能切出 32 段以上
ARTICLE = "\n\n".join(PARA * 6)

e = TTSEngine(MODEL_DIRS, default_variant="custom_voice")
e.warmup()
spk = "sohee"
print("OPT_START", flush=True)

GRID = [(80, 16), (80, 24), (80, 32), (110, 16), (110, 24), (160, 16)]
results = []

for max_chars, bs in GRID:
    segs = build_segments(ARTICLE, max_chars=max_chars)
    texts = [s["text"] for s in segs][:bs]
    if len(texts) < bs:
        print("OPT_SKIP " + json.dumps({"max_chars": max_chars, "batch": bs,
                                        "available": len(texts)}), flush=True)
        continue
    avg_chars = round(sum(len(t) for t in texts) / len(texts))
    try:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        wavs, sr = e.synth_batch(texts, mode="custom", language="korean",
                                 speaker=spk, instruct="차분하고 자연스러운 톤으로")
        el = time.time() - t0
        audio = sum(len(np.asarray(x).flatten()) / sr for x in wavs)
        rtf = el / audio
        peak = torch.cuda.max_memory_allocated() / 1e9
        r = {"max_chars": max_chars, "avg_chars": avg_chars, "batch": bs,
             "gen_s": round(el, 1), "audio_s": round(audio, 1),
             "avg_seg_s": round(audio / bs, 1),
             "rtf": round(rtf, 3), "vram_peak_gb": round(peak, 2)}
        print("OPT " + json.dumps(r), flush=True)
        results.append(r)
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        print("OPT " + json.dumps({"max_chars": max_chars, "batch": bs, "oom": True}), flush=True)
        torch.cuda.empty_cache()
    except Exception as ex:
        print("OPT " + json.dumps({"max_chars": max_chars, "batch": bs,
                                   "error": str(ex)[:150]}), flush=True)
        torch.cuda.empty_cache()

if results:
    best = min(results, key=lambda r: r["rtf"])
    print("OPT_BEST " + json.dumps(best), flush=True)
    for n in (100, 200):
        print("OPT_CAPACITY " + json.dumps({
            "cfg": f"chars{best['max_chars']}_batch{best['batch']}",
            "clips": n, "gpu_h": round(n * 480 * best["rtf"] / 3600, 1)}), flush=True)
print("OPT_DONE", flush=True)
