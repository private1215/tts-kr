# -*- coding: utf-8 -*-
"""第二批-任务A 第1步：从千问整篇产物切「开头片段」作 Cosy 参考，并 ASR 出 ref_text。

关键点（不许偷懒的地方）：
  · 参考音频必须是**千问整篇产物的开头片段**，不是案例原文参考音色；
  · ref_text 必须是**该片段的实际转写**（Whisper-small / CPU / int8），
    不是拿案例全文前几句冒充；
  · 片段起点做 VAD 式微调：跳过开头的静音/爆音，但不跨越到句中。

用法（CPU 为主，不占 GPU）：
    python -u tools/b2_chain_ref.py --tag female
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, "/mnt/tts/projects/qwen3-tts")

STATE = "/mnt/tts/diag/b2-20261013"

# 千问整篇产物（裸推理链路，优先选它作「打底」：不经线上后处理，
# 参考片段里的音色/韵律就是千问模型自己的输出，串联语义最干净）
SRC = {
    "female": "/mnt/tts/diag/longsubmit-20261013/Q2.wav",   # ce1a3fda551a 女
    "male": "/mnt/tts/diag/longsubmit-20261013/Q1.wav",     # d172b8c4efb3 男
}

CLIP_SEC = 13.0


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def speech_onset(x, sr, frame=0.02):
    """返回首个能量超过阈值的帧起点（秒）。用于跳过开头静音。"""
    n = int(frame * sr)
    nf = len(x) // n
    if nf == 0:
        return 0.0
    env = np.abs(x[:nf * n]).reshape(nf, n).max(axis=1)
    peak = float(env.max())
    if peak <= 0:
        return 0.0
    thr = max(0.02 * peak, 1e-4)
    hit = np.nonzero(env > thr)[0]
    if len(hit) == 0:
        return 0.0
    return float(hit[0]) * n / sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["female", "male"])
    ap.add_argument("--clip-sec", type=float, default=CLIP_SEC)
    a = ap.parse_args()

    os.makedirs(STATE, exist_ok=True)
    src = SRC[a.tag]
    x, sr = sf.read(src, dtype="float32", always_2d=False)
    if getattr(x, "ndim", 1) > 1:
        x = x.mean(axis=1)

    onset = speech_onset(x, sr)
    start = max(0.0, onset - 0.10)
    end = min(len(x) / sr, start + a.clip_sec)
    seg = x[int(start * sr):int(end * sr)]

    clip = os.path.join(STATE, "ref_%s_head.wav" % a.tag)
    sf.write(clip, seg, sr, subtype="PCM_16")

    from app.asr_check import transcribe, MODEL_ID
    t0 = time.time()
    hyp = transcribe(clip, "ko")
    asr_sec = round(time.time() - t0, 2)

    rec = {
        "tag": a.tag,
        "src_wav": src,
        "src_sha256": sha(src),
        "src_seconds": round(len(x) / sr, 3),
        "src_sr": sr,
        "speech_onset_sec": round(onset, 3),
        "clip_start_sec": round(start, 3),
        "clip_end_sec": round(end, 3),
        "clip_seconds": round(len(seg) / sr, 3),
        "clip_wav": clip,
        "clip_sha256": sha(clip),
        "asr_model": MODEL_ID,
        "asr_device": "cpu/int8",
        "asr_seconds": asr_sec,
        "ref_text": hyp,
        "ref_text_len": len(hyp),
        "ref_text_sha256": hashlib.sha256(hyp.encode("utf-8")).hexdigest(),
    }
    print("[%s] 片段 %.2fs~%.2fs (%.2fs)  onset=%.2fs" %
          (a.tag, start, end, len(seg) / sr, onset), flush=True)
    print("  ASR(%ss): %s" % (asr_sec, hyp), flush=True)
    json.dump(rec, open(os.path.join(STATE, "ref_%s.json" % a.tag), "w"),
              ensure_ascii=False, indent=1)
    print("  -> ref_%s.json" % a.tag, flush=True)


if __name__ == "__main__":
    main()
