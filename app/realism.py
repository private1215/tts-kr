# -*- coding: utf-8 -*-
"""真实场景测试：用 160 字长片段（约 30 秒音频）测不同批量下的 RTF 与显存。

短句扫描的结果不能直接套用到长片段——显存随"批量 x 片段长度"增长。
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

ARTICLE = """
요즘 인공지능 기술은 정말 빠르게 발전하고 있습니다. 특히 음성 합성 분야에서는 사람의 목소리와 거의 구별하기 어려운 수준까지 도달했습니다. 하지만 아직도 긴 문장을 읽을 때는 어딘가 어색한 느낌이 남아 있습니다.

그 이유는 기계가 문장의 감정과 호흡을 균일하게 처리하기 때문입니다. 사람은 문장을 읽을 때 의미 단위마다 속도를 바꾸고, 중요한 부분에서 잠시 멈춥니다. 이런 미세한 변화가 모여서 자연스러운 말하기가 완성되는 것이죠.

그래서 우리는 긴 원고를 한 번에 읽히는 대신, 의미 단위로 나누어 각각의 호흡을 부여합니다. 이렇게 하면 전체적으로 훨씬 자연스러운 결과를 얻을 수 있습니다. 실제로 들어보면 차이가 분명합니다.

한국어는 특히 문장 끝의 억양이 중요합니다. 같은 문장이라도 끝을 올리면 질문이 되고, 내리면 단정적인 표현이 됩니다. 이런 미묘한 차이를 살리는 것이 자연스러움의 핵심입니다.

또한 한국어에는 한자어와 외래어가 많습니다. 이런 단어들을 어떻게 발음하느냐에 따라서도 전체적인 인상이 달라집니다. 원고를 넣기 전에 미리 점검해 두면 좋습니다.

숫자나 단위가 들어간 문장도 주의가 필요합니다. 쉰 퍼센트라고 읽어야 할지, 오십 퍼센트라고 읽어야 할지는 문맥에 따라 다릅니다. 이런 부분을 미리 정리해 두면 훨씬 자연스럽습니다.

결국 좋은 음성을 만들기 위해서는 모델의 성능도 중요하지만, 원고를 어떻게 준비하느냐도 매우 중요합니다. 두 가지가 함께 맞물려야 비로소 자연스러운 결과가 나옵니다.

앞으로는 이런 부분들이 점점 더 자동화될 것입니다. 하지만 지금 당장은 사람이 직접 확인하고 조정하는 과정이 필요합니다. 그것이 품질을 보장하는 가장 확실한 방법입니다.
"""

e = TTSEngine(MODEL_DIRS, default_variant="custom_voice")
e.warmup()
spk = "sohee"

segs = build_segments(ARTICLE, max_chars=160)
texts = [s["text"] for s in segs]
print("REAL_START", json.dumps({"segments": len(texts),
                                "avg_chars": round(sum(len(t) for t in texts) / len(texts))}),
      flush=True)

# 先跑一个批次摸清单段音频长度
w, sr = e.synth_batch(texts[:2], mode="custom", language="korean",
                      speaker=spk, instruct="차분하고 자연스러운 톤으로")
per_seg = sum(len(np.asarray(x).flatten()) / sr for x in w) / len(w)
print("REAL_SEGLEN", json.dumps({"avg_audio_s_per_segment": round(per_seg, 1)}), flush=True)

results = []
for bs in (4, 6, 8, 12, 16):
    batch = texts[:bs]
    try:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        wavs, sr = e.synth_batch(batch, mode="custom", language="korean",
                                 speaker=spk, instruct="차분하고 자연스러운 톤으로")
        el = time.time() - t0
        audio = sum(len(np.asarray(x).flatten()) / sr for x in wavs)
        rtf = el / audio
        peak = torch.cuda.max_memory_allocated() / 1e9
        r = {"batch": bs, "gen_s": round(el, 1), "audio_s": round(audio, 1),
             "rtf": round(rtf, 3), "vram_peak_gb": round(peak, 2)}
        print("REAL " + json.dumps(r), flush=True)
        results.append(r)
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        print("REAL " + json.dumps({"batch": bs, "oom": True}), flush=True)
        torch.cuda.empty_cache()
        break
    except Exception as ex:
        print("REAL " + json.dumps({"batch": bs, "error": str(ex)[:150]}), flush=True)
        torch.cuda.empty_cache()
        break

ok = [r for r in results if "rtf" in r]
if ok:
    best = min(ok, key=lambda r: r["rtf"])
    print("REAL_BEST " + json.dumps(best), flush=True)
    for n in (100, 200):
        print("REAL_CAPACITY " + json.dumps({
            "batch": best["batch"], "clips": n,
            "gpu_h": round(n * 480 * best["rtf"] / 3600, 1)}), flush=True)
print("REAL_DONE", flush=True)
