# -*- coding: utf-8 -*-
"""第二批-任务A 第2步：真串联 —— 以「千问整篇开头片段」为参考，让 CosyVoice3
零样本克隆并念**同一篇完整稿件**。

与「拼接/混音」的界线：本脚本只向 cosy 提交 (ref_wav=千问片段, ref_text=该片段ASR,
text=整篇稿件)，产出音频 100% 由 CosyVoice3 生成，不存在任何把千问音频
接进来或混进来的操作。

情绪：显式设置 instruct（非 neutral），值记入 json 留痕。

用法（GPU 串行，一次一条）：
    CUDA_VISIBLE_DEVICES=0 python -u tools/b2_chain_cosy.py \
        --tag female --case ce1a3fda551a --name C1 --instruct excited
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
STATE = "/mnt/tts/diag/b2-20261013"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["female", "male"])
    ap.add_argument("--case", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--instruct", default="excited")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-chars", type=int, default=320)
    ap.add_argument("--timeout", type=float, default=3600.0)
    a = ap.parse_args()

    ref = json.load(open(os.path.join(STATE, "ref_%s.json" % a.tag),
                         encoding="utf-8"))
    txt = open(os.path.join(CASES, a.case, "full.txt"), encoding="utf-8").read()

    rec = {
        "name": a.name,
        "route": "串联克隆-千问片段打底+Cosy整篇",
        "engine": "cosyvoice3",
        "case_id": a.case,
        "tag": a.tag,
        "base_engine": "qwen（整篇产物开头片段，仅作参考音频，不参与混音/拼接）",
        "text_chars": len(txt),
        "text_sha256": hashlib.sha256(txt.encode("utf-8")).hexdigest(),
        "ref_wav": ref["clip_wav"],
        "ref_wav_sha256": ref["clip_sha256"],
        "ref_source_wav": ref["src_wav"],
        "ref_source_sha256": ref["src_sha256"],
        "ref_interval_sec": [ref["clip_start_sec"], ref["clip_end_sec"]],
        "ref_seconds": ref["clip_seconds"],
        "ref_text": ref["ref_text"],
        "ref_text_len": ref["ref_text_len"],
        "ref_text_sha256": ref["ref_text_sha256"],
        "ref_text_source": "Whisper-small CPU int8 转写该片段（asr_model=%s）"
                           % ref["asr_model"],
        "params": {"instruct": a.instruct, "speed": 1.0, "seed": a.seed,
                   "max_chars": a.max_chars, "timeout": a.timeout},
        "service_url": os.environ.get("TTS_COSY_URL", "http://127.0.0.1:8100"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    print("[%s] %s 文案%d字  参考=%s (%.2fs, 来自 %s)  instruct=%s" %
          (a.name, a.case, len(txt), ref["clip_wav"], ref["clip_seconds"],
           os.path.basename(ref["src_wav"]), a.instruct), flush=True)
    print("  ref_text: %s" % ref["ref_text"], flush=True)

    from app.cosy_client import synth, health, COSY_MAX_CHARS
    rec["COSY_MAX_CHARS_default"] = COSY_MAX_CHARS
    ok, hv = health()
    rec["service_health_before"] = {"ok": ok, "info": hv}
    print("  服务健康:", ok, hv.get("status"), flush=True)

    t0 = time.time()
    audio, sr, meta = synth(text=txt, ref_wav=ref["clip_wav"],
                            ref_text=ref["ref_text"], instruct=a.instruct,
                            speed=1.0, seed=a.seed, max_chars=a.max_chars,
                            timeout=a.timeout)
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
    rec["wav_sha256"] = sha(dst)
    print("  -> %s  %.1fs @%dHz  gen=%.1fs" %
          (dst, len(audio) / sr, sr, rec["gen_seconds"]), flush=True)
    json.dump(rec, open(os.path.join(STATE, a.name + ".json"), "w"),
              ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
