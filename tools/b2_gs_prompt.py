# -*- coding: utf-8 -*-
"""第二批-任务B：为 GS 裁出 3~10 秒提示音频，并 ASR 出对应的 prompt_text。

GS 的 TTS 前端硬校验「参考音频 3~10 秒」，现役 voices.json 里的音色是 15 秒，
直接塞会抛 OSError。这里裁开头 9 秒（跳过静音），并用 Whisper-small CPU 转写，
保证 prompt_text 与这段音频**真实对应**，不拿整段的稿子冒充。

用法（CPU，不占 GPU）：
    python -u tools/b2_gs_prompt.py --voice SOLA_female --tag sola
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

VOICES = "/mnt/tts/data/voices"
STATE = "/mnt/tts/diag/b2-20261013"
CLIP_SEC = 9.0


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def speech_onset(x, sr, frame=0.02):
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
    return float(hit[0]) * n / sr if len(hit) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True, help="voices.json 里的音色名")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--clip-sec", type=float, default=CLIP_SEC)
    a = ap.parse_args()

    os.makedirs(STATE, exist_ok=True)
    src = os.path.join(VOICES, a.voice + ".wav")
    x, sr = sf.read(src, dtype="float32", always_2d=False)
    if getattr(x, "ndim", 1) > 1:
        x = x.mean(axis=1)

    onset = speech_onset(x, sr)
    start = max(0.0, onset - 0.08)
    end = min(len(x) / sr, start + a.clip_sec)
    seg = x[int(start * sr):int(end * sr)]
    clip = os.path.join(STATE, "gsprompt_%s.wav" % a.tag)
    sf.write(clip, seg, sr, subtype="PCM_16")

    from app.asr_check import transcribe, MODEL_ID
    t0 = time.time()
    hyp = transcribe(clip, "ko")

    rec = {
        "voice": a.voice,
        "src_wav": src,
        "src_sha256": sha(src),
        "src_seconds": round(len(x) / sr, 3),
        "speech_onset_sec": round(onset, 3),
        "clip_start_sec": round(start, 3),
        "clip_end_sec": round(end, 3),
        "clip_seconds": round(len(seg) / sr, 3),
        "clip_wav": clip,
        "clip_sha256": sha(clip),
        "asr_model": MODEL_ID,
        "asr_seconds": round(time.time() - t0, 2),
        "ref_text": hyp,
        "ref_text_len": len(hyp),
    }
    print("[%s] %s  %.2fs~%.2fs (%.2fs)" %
          (a.tag, a.voice, start, end, len(seg) / sr), flush=True)
    print("  ASR: %s" % hyp, flush=True)
    json.dump(rec, open(os.path.join(STATE, "gsprompt_%s.json" % a.tag), "w"),
              ensure_ascii=False, indent=1)
    print("  -> gsprompt_%s.json" % a.tag, flush=True)


if __name__ == "__main__":
    main()
