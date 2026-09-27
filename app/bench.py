# -*- coding: utf-8 -*-
"""Qwen3-TTS 基准测试：加载耗时、显存、韩语 RTF。

RTF 与音频长度基本无关，用短样本即可测准，再外推到 8 分钟场景。
"""
import os
import sys
import time
import json

import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_engine import TTSEngine
from pipeline import build_segments

MODEL_DIR = "/mnt/tts/models"
MODEL_DIRS = {
    "custom_voice": os.path.join(MODEL_DIR, "Qwen3-TTS-12Hz-1.7B-CustomVoice"),
    "base": os.path.join(MODEL_DIR, "Qwen3-TTS-12Hz-1.7B-Base"),
}

# 8 句韩语短句，每句约 5-10 秒音频
SENTS = [
    "요즘 인공지능 기술은 정말 빠르게 발전하고 있습니다.",
    "특히 음성 합성 분야에서는 사람의 목소리와 거의 구별하기 어려운 수준까지 도달했습니다.",
    "하지만 긴 문장을 읽을 때는 아직도 어딘가 어색한 느낌이 남아 있습니다.",
    "그 이유는 기계가 문장의 감정과 호흡을 균일하게 처리하기 때문입니다.",
    "사람은 의미 단위마다 속도를 바꾸고 중요한 부분에서 잠시 멈춥니다.",
    "이런 미세한 변화가 모여서 자연스러운 말하기가 완성됩니다.",
    "그래서 긴 원고는 의미 단위로 나누어 각각의 호흡을 부여합니다.",
    "이렇게 하면 전체적으로 훨씬 자연스러운 결과를 얻을 수 있습니다.",
]

result = {}
log_lines = []


def emit(tag, obj):
    line = tag + " " + json.dumps(obj, ensure_ascii=False)
    print(line, flush=True)
    log_lines.append(line)


e = TTSEngine(MODEL_DIRS, default_variant="custom_voice")
t0 = time.time()
e.warmup()
result["load_seconds"] = round(time.time() - t0, 1)
result["variant"] = e.variant
result["languages"] = e.languages
result["speakers"] = e.speakers
result["vram_after_load_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
emit("INFO", {k: v for k, v in result.items() if k != "speakers"})

spk = "Sohee"
for s in e.speakers:
    if "sohee" in str(s).lower():
        spk = s
        break
emit("INFO", {"speaker": spk})


def bench(name, texts):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    wavs, sr = e.synth_batch(texts, mode="custom", language="Korean",
                             speaker=spk, instruct="차분하고 자연스러운 톤으로")
    el = time.time() - t0
    durs = [len(np.asarray(w).flatten()) / sr for w in wavs]
    audio = sum(durs)
    rtf = el / audio if audio > 0 else 0
    r = {"case": name, "segments": len(texts),
         "gen_s": round(el, 1), "audio_s": round(audio, 1),
         "rtf": round(rtf, 3),
         "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    emit("BENCH", r)
    result.setdefault("bench", []).append(r)
    return r


bench("warmup_1", [SENTS[0]])
bench("single_1", [SENTS[1]])
bench("batch_4", SENTS[:4])
bench("batch_8", SENTS)

# 端到端：分段 + 批量 + 韵律拼接
long_text = "\n\n".join(SENTS)
segs = build_segments(long_text, max_chars=160)
emit("INFO", {"long_segments": len(segs)})
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
audio, sr, stats = e.synth_long(long_text, mode="custom", language="Korean",
                                speaker=spk, instruct="차분하고 자연스러운 톤으로",
                                batch_size=8)
stats["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
emit("LONG", stats)
result["long"] = stats

os.makedirs("/mnt/tts/output", exist_ok=True)
sf.write("/mnt/tts/output/bench_sample.wav", audio, sr)
emit("INFO", {"sample": "/mnt/tts/output/bench_sample.wav",
              "dur_s": round(len(audio) / sr, 1)})

# 产能外推
rtf = result["bench"][-1]["rtf"]
for n in (100, 200):
    gpu_h = round(n * 480 * rtf / 3600, 1)
    emit("CAPACITY", {"clips_per_day": n, "gpu_hours_needed": gpu_h,
                      "feasible": gpu_h < 20})
emit("DONE", {"ok": True})

with open("/mnt/tts/projects/qwen3-tts/bench_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
