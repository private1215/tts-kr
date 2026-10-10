# -*- coding: utf-8 -*-
"""多种子 / 多 checkpoint 稳健性验证。

🔴 本项目已经栽过两次跟头（LoRA adapter、句尾下降）：单样本看着有效，
多种子一测就翻车。所以这里一次跑齐，看波动是否大于差异。
"""
import os
import sys
import numpy as np

REPO = "/mnt/tts/projects/gpt-sovits/repo"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))
os.chdir(REPO)

import yaml  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402


def _sf_load(path):
    w, sr = sf.read(path, dtype="float32")
    t = torch.from_numpy(np.asarray(w))
    if t.ndim == 1:
        t = t.unsqueeze(0)
    else:
        t = t.transpose(0, 1)
    return t, sr


torchaudio.load = _sf_load
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

OUT = "/mnt/tts/projects/gpt-sovits/out"
PROMPT = os.path.join(OUT, "prompt_10s.wav")
PTEXT = open(os.path.join(OUT, "prompt_10s.txt"), encoding="utf-8").read().strip()
TEXT = open("/mnt/tts/diag/hh_short.txt", encoding="utf-8").read().strip()
GPT_W = "GPT_SoVITS/pretrained_models/s1v3.ckpt"

CKPTS = sorted([f for f in os.listdir("SoVITS_weights_v2ProPlus")
                if f.endswith(".pth")])
# 取最早 / 中间 / 最后 三个 checkpoint，看有没有过拟合
if len(CKPTS) >= 3:
    CKPTS = [CKPTS[1], CKPTS[len(CKPTS) // 2], CKPTS[-1]]

SEEDS = [42, 2026]


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


def synth(tts, seed, path):
    out = None
    for item in tts.run({
        "text": TEXT, "text_lang": "ko",
        "ref_audio_path": PROMPT, "prompt_text": PTEXT, "prompt_lang": "ko",
        "top_k": 15, "top_p": 1.0, "temperature": 1.0,
        "text_split_method": "cut1", "batch_size": 1, "speed_factor": 1.0,
        "fragment_interval": 0.3, "seed": seed, "parallel_infer": True,
        "repetition_penalty": 1.35, "use_cuda_graph": False,
    }):
        out = item
    sr, audio = out
    audio = np.asarray(audio)
    sf.write(path, audio, sr, subtype="PCM_16")
    print("   ->", os.path.basename(path), "%.1fs" % (len(audio) / sr), flush=True)


def main():
    print("checkpoints:", CKPTS, "seeds:", SEEDS, flush=True)
    for ck in CKPTS:
        tts = TTS(build(os.path.join("SoVITS_weights_v2ProPlus", ck)))
        for sd in SEEDS:
            tag = ck.replace("hh_v2pro_", "").replace(".pth", "")
            synth(tts, sd, os.path.join(OUT, "gs_%s_s%d.wav" % (tag, sd)))
        del tts
        torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
