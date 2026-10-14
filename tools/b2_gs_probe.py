# -*- coding: utf-8 -*-
"""第二批-任务B：GS(GPT-SoVITS) **备份权重**隔离试推理。

铁律（本脚本逐条守着）：
  · 只按绝对路径加载给定的 .pth，**不复制/不移动/不改名**，不写进
    SoVITS_weights_v2ProPlus 顶层目录，因此 pick_weight() 依旧解析不到它；
  · 不改 serve.py、不改任何配置、不重启 gs 服务（端口 8200）、不提升为现役；
  · 产物写到 /mnt/tts/diag/ 下，不进 gpt-sovits/out 的生产目录命名空间；
  · 输出一律标注「来自备份权重、未经验证训练收敛、仅限实验比较」。

用法（GPU 串行，一次一条）：
    CUDA_VISIBLE_DEVICES=0 HF_HOME=/mnt/tts/models/asr HF_HUB_OFFLINE=1 \
      venv/bin/python -u gs_probe.py --weight <abs .pth> --ref <wav> \
      --ref-text "<转写>" --case ce1a3fda551a --mode short --out /path.wav
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
import torchaudio

REPO = "/mnt/tts/projects/gpt-sovits/repo"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))
os.chdir(REPO)


def _sf_load(path):
    w, sr = sf.read(path, dtype="float32")
    t = torch.from_numpy(np.asarray(w))
    if t.ndim == 1:
        t = t.unsqueeze(0)
    else:
        t = t.transpose(0, 1)
    return t, sr


torchaudio.load = _sf_load

GPT_W = "GPT_SoVITS/pretrained_models/s1v3.ckpt"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def short_text(txt, lo=30, hi=60):
    """取开头 30~60 字符，尽量停在句末。"""
    s = txt.strip().replace("\r", "")
    cut = -1
    for i, ch in enumerate(s):
        if ch in ".!?。\n" and i + 1 >= lo:
            cut = i + 1
            break
        if i + 1 >= hi:
            break
    if cut > 0:
        return s[:cut].strip()
    return s[:hi].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True, help="SoVITS .pth 绝对路径")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-text", default="")
    ap.add_argument("--ref-text-file", default="",
                    help="参考音频转写文本文件；给了就优先用它")
    ap.add_argument("--case", required=True)
    ap.add_argument("--mode", choices=["short", "full"], default="short")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", default="")
    a = ap.parse_args()

    if a.ref_text_file:
        a.ref_text = open(a.ref_text_file, encoding="utf-8").read().strip()
    txt = open("/mnt/tts/data/cases/%s/full.txt" % a.case,
               encoding="utf-8").read()
    if a.mode == "short":
        text = short_text(txt)
    else:
        text = txt.strip()

    rec = {
        "mode": a.mode,
        "case_id": a.case,
        "weight_path": a.weight,
        "weight_sha256": sha256(a.weight),
        "weight_bytes": os.path.getsize(a.weight),
        "gpt_weight": GPT_W,
        "ref_wav": a.ref,
        "ref_wav_sha256": sha256(a.ref),
        "ref_text": a.ref_text,
        "ref_text_len": len(a.ref_text),
        "text_chars": len(text),
        "text_head": text[:80],
        "seed": a.seed,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "disclaimer": "备份权重，未经验证训练收敛，仅限实验比较；未提升为现役",
    }
    print("[%s] weight=%s" % (a.mode, os.path.basename(a.weight)), flush=True)
    print("  sha256=%s" % rec["weight_sha256"], flush=True)
    print("  device=%s" % rec["device_name"], flush=True)
    print("  text(%d字): %s" % (len(text), text[:60]), flush=True)

    t_load = time.time()
    try:
        import yaml
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

        def build(sovits_path):
            cfg = yaml.safe_load(open("GPT_SoVITS/configs/tts_infer.yaml"))
            c = cfg.get("custom") or {}
            c["version"] = "v2ProPlus"
            c["device"] = "cuda"
            c["is_half"] = True
            c["t2s_weights_path"] = GPT_W
            c["vits_weights_path"] = sovits_path
            cfg["custom"] = c
            return TTS_Config(cfg)

        tts = TTS(build(a.weight))
        rec["load_seconds"] = round(time.time() - t_load, 2)
        print("  加载成功 %.1fs" % rec["load_seconds"], flush=True)
    except Exception as e:
        rec["status"] = "load_failed"
        rec["error"] = "%s: %s" % (type(e).__name__, e)
        print("  ❌ 加载失败:", rec["error"][:400], flush=True)
        _dump(rec, a)
        return 1

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        out = None
        for item in tts.run({
            "text": text,
            "text_lang": "ko",
            "ref_audio_path": a.ref,
            "prompt_text": a.ref_text,
            "prompt_lang": "ko",
            "top_k": 15,
            "top_p": 1.0,
            "temperature": 1.0,
            "text_split_method": "cut1",
            "batch_size": 1,
            "speed_factor": 1.0,
            "fragment_interval": 0.3,
            "seed": a.seed,
            "parallel_infer": True,
            "repetition_penalty": 1.35,
            "use_cuda_graph": False,
        }):
            out = item
        if out is None:
            raise RuntimeError("推理没有产出音频")
        sr, audio = out
    except Exception as e:
        rec["status"] = "infer_failed"
        rec["error"] = "%s: %s" % (type(e).__name__, e)
        print("  ❌ 推理失败:", rec["error"][:400], flush=True)
        _dump(rec, a)
        return 1

    rec["gen_seconds"] = round(time.time() - t0, 2)
    rec["vram_peak_gib"] = round(
        torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
    audio = np.asarray(audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    sf.write(a.out, audio, sr, subtype="PCM_16")

    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    rms = float(np.sqrt((audio ** 2).mean())) if len(audio) else 0.0
    # 非静音占比：20ms 帧 rms > 1e-4
    n = max(1, int(0.02 * sr))
    nf = len(audio) // n
    env = np.abs(audio[:nf * n]).reshape(nf, n).max(axis=1) if nf else np.zeros(1)
    voiced = float((env > 1e-4).mean()) if nf else 0.0

    rec.update({
        "status": "done",
        "out": a.out,
        "sr": int(sr),
        "audio_seconds": round(len(audio) / sr, 2),
        "peak": round(peak, 5),
        "rms": round(rms, 5),
        "voiced_ratio": round(voiced, 4),
        "cps": round(len(text) / (len(audio) / sr), 3) if len(audio) else None,
        "wav_sha256": sha256(a.out),
        "checks": {
            "可解码": True,
            "非静音": bool(peak > 1e-3 and rms > 1e-4),
            "时长合理": None,   # 由调用方按 cps 判定后回填
        },
    })
    print("  -> %s  %.1fs @%dHz  peak=%.3f rms=%.4f voiced=%.2f gen=%.1fs "
          "vram_peak=%.2fGiB" %
          (a.out, len(audio) / sr, sr, peak, rms, voiced,
           rec["gen_seconds"], rec["vram_peak_gib"]), flush=True)
    _dump(rec, a)
    return 0


def _dump(rec, a):
    if a.report:
        json.dump(rec, open(a.report, "w"), ensure_ascii=False, indent=1)
        print("  ->", a.report, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
