# -*- coding: utf-8 -*-
"""裸推理-Qwen：整篇长文案直接交给 TTSEngine.synth_long，不叠加任何后处理。

与「线上完整链-Qwen」的差异（本轮就是要留这个对照）：
  · 不经服务端关口：无 ASR 回检重试、无慢速修复、无按预估时长的自适应分块
  · 不经 PBA 停顿对齐 / EQ / PROSO 韵律移植 / 呼吸注入 / 响度闭环 / peakfit
  · 不传 voice_match（保真增强目标曲线），这项目线上开启、裸推理关闭
  · 模型同一份：Qwen3-TTS-12Hz-1.7B-Base（clone 模式的变体与线上一致）

用法（串行，一次一条）：
    CUDA_VISIBLE_DEVICES=1 python -u tools/longsubmit_bareqwen.py \
        --case d172b8c4efb3 --voice EL_male --name Q1
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")
sys.path.insert(0, "/mnt/tts/projects/qwen3-tts/app")   # tts_engine / pipeline 在 app/

CASES = "/mnt/tts/data/cases"
VOICES = "/mnt/tts/data/voices/voices.json"
STATE = "/mnt/tts/diag/longsubmit-20261013"

BASE = os.environ.get("TTS_BASE", "").strip() or "/mnt/tts"
MODEL_DIRS = {
    "custom_voice": os.path.join(BASE, "models", "Qwen3-TTS-12Hz-1.7B-CustomVoice"),
    "base": os.path.join(BASE, "models", "Qwen3-TTS-12Hz-1.7B-Base"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-chars", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()

    os.makedirs(STATE, exist_ok=True)
    txt = open(os.path.join(CASES, a.case, "full.txt"), encoding="utf-8").read()
    v = json.load(open(VOICES, encoding="utf-8"))[a.voice]
    pwav, ptx = v["path"], (v.get("ref_text") or "")

    rec = {
        "name": a.name, "route": "裸推理-Qwen", "engine": "qwen",
        "case_id": a.case, "voice": a.voice,
        "text_chars": len(txt),
        "text_sha256": hashlib.sha256(txt.encode("utf-8")).hexdigest(),
        "ref_wav": pwav,
        "ref_wav_sha256": hashlib.sha256(open(pwav, "rb").read()).hexdigest(),
        "ref_text_sha256": hashlib.sha256(ptx.encode("utf-8")).hexdigest(),
        "ref_text_len": len(ptx),
        "model_path": MODEL_DIRS["base"],
        "params": {"mode": "clone", "language": "Korean", "seed": a.seed,
                   "max_chars": a.max_chars, "batch_size": a.batch_size,
                   "instruct": "", "speed_note": "Qwen 无 speed 入参，变速由后端 atempo 实现，本条未叠"},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": torch.__version__,
    }
    print("[%s] 文案 %d 字 模型=%s" % (a.name, len(txt), MODEL_DIRS["base"]),
          flush=True)

    from tts_engine import TTSEngine
    e = TTSEngine(MODEL_DIRS, default_variant="base")
    t_warm = time.time()
    e.warmup()
    rec["warmup_seconds"] = round(time.time() - t_warm, 2)
    print("  warmup %.1fs" % rec["warmup_seconds"], flush=True)

    ref_audio = sf.read(pwav, dtype="float32")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    audio, sr, st = e.synth_long(
        text=txt, mode="clone", language="Korean", speaker=None,
        instruct="", ref_audio=ref_audio, ref_text=ptx, ref_key=a.voice,
        max_chars=a.max_chars, batch_size=a.batch_size, seed=a.seed,
        voice_match=None)
    rec["gen_seconds"] = round(time.time() - t0, 2)
    if torch.cuda.is_available():
        rec["vram_peak_gib"] = round(
            torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
    audio = np.asarray(audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    dst = os.path.join(STATE, a.name + ".wav")
    sf.write(dst, audio, sr, subtype="PCM_16")
    rec["status"] = "done"
    rec["out"] = dst
    rec["sr"] = sr
    rec["audio_seconds"] = round(len(audio) / sr, 2)
    rec["wav_sha256"] = hashlib.sha256(open(dst, "rb").read()).hexdigest()
    rec["synth_long_stats"] = {k: st.get(k) for k in
                               ("segments", "chars", "gen_seconds",
                                "audio_seconds", "cps", "max_chars",
                                "batch_size", "clone_prompt_tokens")}
    print("  -> %s  %.1fs @%dHz gen=%.1fs vram_peak=%sGiB" %
          (dst, len(audio) / sr, sr, rec["gen_seconds"],
           rec.get("vram_peak_gib")), flush=True)
    json.dump(rec, open(os.path.join(STATE, a.name + ".json"), "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
