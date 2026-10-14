# -*- coding: utf-8 -*-
"""裸推理-Cosy：整篇长文案直接交给 cosy_client.synth，不叠加任何后处理。

与线上链的区别（本轮就是要留这个对照）：
  · 不经过 PBA 停顿对齐 / 慢速修复 / EQ / PROSO / 呼吸 / 响度闭环 / peakfit
  · 不经过服务端按「预估时长」的分块，用的是 cosy_client 的 max_chars=320 切分
  · 参考音频用 voices.json 里的现役 EL_* （与线上链提交同源，保证只差「链路」）

用法（必须串行，一次一条）：
    CUDA_VISIBLE_DEVICES=1 python -u tools/longsubmit_cosy.py \
        --case d172b8c4efb3 --voice EL_male --name M3
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

CASES = "/mnt/tts/data/cases"
VOICES = "/mnt/tts/data/voices/voices.json"
STATE = "/mnt/tts/diag/longsubmit-20261013"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-chars", type=int, default=320)
    ap.add_argument("--timeout", type=float, default=3600.0)
    a = ap.parse_args()

    os.makedirs(STATE, exist_ok=True)
    txt = open(os.path.join(CASES, a.case, "full.txt"), encoding="utf-8").read()
    v = json.load(open(VOICES, encoding="utf-8"))[a.voice]
    pwav, ptx = v["path"], (v.get("ref_text") or "")

    rec = {
        "name": a.name, "route": "裸推理-Cosy", "engine": "cosyvoice3",
        "case_id": a.case, "voice": a.voice,
        "text_chars": len(txt),
        "text_sha256": hashlib.sha256(txt.encode("utf-8")).hexdigest(),
        "ref_wav": pwav,
        "ref_wav_sha256": hashlib.sha256(open(pwav, "rb").read()).hexdigest(),
        "ref_text_sha256": hashlib.sha256(ptx.encode("utf-8")).hexdigest(),
        "ref_text_len": len(ptx),
        "params": {"instruct": "neutral", "speed": 1.0, "seed": a.seed,
                   "max_chars": a.max_chars, "timeout": a.timeout},
        "service_url": os.environ.get("TTS_COSY_URL", "http://127.0.0.1:8100"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    print("[%s] 文案 %d 字  参考=%s  ref_text=%d字" %
          (a.name, len(txt), pwav, len(ptx)), flush=True)

    from app.cosy_client import synth, health, COSY_MAX_CHARS
    rec["COS Y_MAX_CHARS_default"] = COSY_MAX_CHARS
    ok, hv = health()
    rec["service_health_before"] = {"ok": ok, "info": hv}
    print("  服务健康:", ok, hv, flush=True)

    t0 = time.time()
    audio, sr, meta = synth(text=txt, ref_wav=pwav, ref_text=ptx,
                            instruct="neutral", speed=1.0, seed=a.seed,
                            max_chars=a.max_chars, timeout=a.timeout)
    rec["gen_seconds"] = round(time.time() - t0, 2)
    rec["meta"] = meta
    if audio is None:
        rec["status"] = "failed"
        rec["error"] = meta.get("error")
        print("  ❌ 失败:", meta.get("error"), flush=True)
        json.dump(rec, open(os.path.join(STATE, a.name + ".json"), "w"),
                  ensure_ascii=False, indent=1)
        return 1

    audio = np.asarray(audio, dtype="float32")
    ok2, hv2 = health()
    rec["service_health_after"] = {"ok": ok2, "info": hv2}
    rec["vram_gib_after"] = hv2.get("vram_gib")
    dst = os.path.join(STATE, a.name + ".wav")
    sf.write(dst, audio, sr, subtype="PCM_16")
    rec["status"] = "done"
    rec["out"] = dst
    rec["sr"] = sr
    rec["audio_seconds"] = round(len(audio) / sr, 2)
    rec["wav_sha256"] = hashlib.sha256(open(dst, "rb").read()).hexdigest()
    print("  -> %s  %.1fs @%dHz  gen=%.1fs" %
          (dst, len(audio) / sr, sr, rec["gen_seconds"]), flush=True)
    json.dump(rec, open(os.path.join(STATE, a.name + ".json"), "w"),
              ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
