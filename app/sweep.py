# -*- coding: utf-8 -*-
"""批量大小扫描：找出 RTF / 显存 的最佳平衡点。"""
import os
import sys
import time
import json

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_engine import TTSEngine

# 换服务器时设 TTS_BASE 即可整体迁移，不设则等价于 /mnt/tts
BASE = os.environ.get("TTS_BASE", "").strip() or "/mnt/tts"
MODEL_DIR = os.path.join(BASE, "models")
MODEL_DIRS = {
    "custom_voice": os.path.join(MODEL_DIR, "Qwen3-TTS-12Hz-1.7B-CustomVoice"),
    "base": os.path.join(MODEL_DIR, "Qwen3-TTS-12Hz-1.7B-Base"),
}

BASE = [
    "요즘 인공지능 기술은 정말 빠르게 발전하고 있습니다.",
    "특히 음성 합성 분야에서는 사람의 목소리와 거의 구별하기 어려운 수준까지 도달했습니다.",
    "하지만 긴 문장을 읽을 때는 아직도 어딘가 어색한 느낌이 남아 있습니다.",
    "그 이유는 기계가 문장의 감정과 호흡을 균일하게 처리하기 때문입니다.",
    "사람은 의미 단위마다 속도를 바꾸고 중요한 부분에서 잠시 멈춥니다.",
    "이런 미세한 변화가 모여서 자연스러운 말하기가 완성됩니다.",
    "그래서 긴 원고는 의미 단위로 나누어 각각의 호흡을 부여합니다.",
    "이렇게 하면 전체적으로 훨씬 자연스러운 결과를 얻을 수 있습니다.",
]


def texts_for(n):
    out = []
    while len(out) < n:
        out.extend(BASE)
    return out[:n]


e = TTSEngine(MODEL_DIRS, default_variant="custom_voice")
e.warmup()
spk = "sohee"
print("SWEEP_START", flush=True)

results = []
for bs in (8, 12, 16, 20, 24, 32):
    texts = texts_for(bs)
    try:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        wavs, sr = e.synth_batch(texts, mode="custom", language="korean",
                                 speaker=spk, instruct="차분하고 자연스러운 톤으로")
        el = time.time() - t0
        audio = sum(len(np.asarray(w).flatten()) / sr for w in wavs)
        rtf = el / audio
        peak = torch.cuda.max_memory_allocated() / 1e9
        r = {"batch": bs, "gen_s": round(el, 1), "audio_s": round(audio, 1),
             "rtf": round(rtf, 3), "vram_peak_gb": round(peak, 2)}
        print("SWEEP " + json.dumps(r), flush=True)
        results.append(r)
    except torch.cuda.OutOfMemoryError:
        print("SWEEP " + json.dumps({"batch": bs, "oom": True}), flush=True)
        torch.cuda.empty_cache()
        break
    except Exception as ex:
        print("SWEEP " + json.dumps({"batch": bs, "error": str(ex)[:120]}), flush=True)
        torch.cuda.empty_cache()
        break

best = min(results, key=lambda r: r["rtf"]) if results else None
print("SWEEP_BEST " + json.dumps(best), flush=True)
if best:
    for n in (100, 200):
        print("CAPACITY " + json.dumps({
            "batch": best["batch"], "clips": n,
            "gpu_h": round(n * 480 * best["rtf"] / 3600, 1)}), flush=True)
print("SWEEP_DONE", flush=True)
