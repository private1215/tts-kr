# -*- coding: utf-8 -*-
"""GPT-SoVITS 推理：用韩华案例同文案生成，和现有 Qwen3-TTS 做对照。

两种模式都跑，这样才知道"微调到底带来多少"：
  A. finetune  —— 用我们微调过的 SoVITS v2ProPlus
  B. zeroshot  —— 用官方预训练权重，一点都不微调

参照音频（prompt）**必须取自测试区**（案例前 60 秒，未进训练集），
否则等于拿训练素材当提示，结果不诚实。

用法：
    python gs_infer.py                 # 两个模式都跑
    python gs_infer.py finetune        # 只跑微调版
"""
import os
import sys
import subprocess
import numpy as np

REPO = "/mnt/tts/projects/gpt-sovits/repo"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))
os.chdir(REPO)

import yaml  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402


# 🔴 这台服务器的 torchaudio 一律走 torchcodec，而它的 .so 要 FFmpeg4 的
#    libavcodec.so.58 和 libnppicc.so.12，系统都没有（只有 FFmpeg6 的 .so.60）。
#    TTS.py 两处用 torchaudio.load 读参照音频，直接换掉即可，不动上游代码。
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

CASE = "/mnt/tts/data/cases/c3117b7b0fb1"
MERGED = os.path.join(CASE, "merged.wav")
OUT = "/mnt/tts/projects/gpt-sovits/out"
os.makedirs(OUT, exist_ok=True)

TEXT = open("/mnt/tts/diag/hh_short.txt", encoding="utf-8").read().strip()
GPT_W = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
PRETRAINED_SOVITS = "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth"
TRAINED_SOVITS = sorted(
    [f for f in os.listdir("SoVITS_weights_v2ProPlus") if f.endswith(".pth")]
)[-1] if os.path.isdir("SoVITS_weights_v2ProPlus") else None

# 提示音频：测试区里取 10 秒（5s ~ 15s），保证没被训练过
PROMPT = os.path.join(OUT, "prompt_10s.wav")
PROMPT_TXT = os.path.join(OUT, "prompt_10s.txt")


def make_prompt():
    if os.path.exists(PROMPT) and os.path.exists(PROMPT_TXT):
        return open(PROMPT_TXT, encoding="utf-8").read().strip()
    x, sr = sf.read(MERGED, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    seg = x[int(5 * sr):int(15 * sr)]
    sf.write(PROMPT, seg, sr, subtype="PCM_16")
    # 用 faster-whisper 转写提示文本（gs 环境里已装）
    os.environ.setdefault("HF_HOME", "/mnt/tts/models/asr")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from faster_whisper import WhisperModel
    m = WhisperModel("medium", device="cuda", compute_type="int8")
    segs, _ = m.transcribe(PROMPT, language="ko", beam_size=1)
    txt = " ".join(s.text for s in segs).strip()
    open(PROMPT_TXT, "w", encoding="utf-8").write(txt)
    return txt


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


def synth(tts, text, prompt, prompt_text, path, seed=42):
    # ⚠️ 这个版本 TTS.run 是**生成器**（内部 yield），不是直接返回 tuple，
    #    必须迭代；最后一包才是完整音频。
    out = None
    for item in tts.run({
        "text": text,
        "text_lang": "ko",
        "ref_audio_path": prompt,
        "prompt_text": prompt_text,
        "prompt_lang": "ko",
        "top_k": 15,
        "top_p": 1.0,
        "temperature": 1.0,
        "text_split_method": "cut1",
        "batch_size": 1,
        "speed_factor": 1.0,
        "fragment_interval": 0.3,
        "seed": seed,
        "parallel_infer": True,
        "repetition_penalty": 1.35,
        "use_cuda_graph": False,          # 这台机器上 cuda_graph 捕获会卡住
    }):
        out = item
    if out is None:
        raise RuntimeError("推理没有产出音频")
    sr, audio = out
    audio = np.asarray(audio)
    sf.write(path, audio, sr, subtype="PCM_16")
    print("   ->", path, "%.1fs @%dHz" % (len(audio) / sr, sr), flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    ptext = make_prompt()
    print("提示文本:", ptext, flush=True)
    print("目标文本(%d字): %s" % (len(TEXT), TEXT[:40] + "..."), flush=True)
    print("微调权重:", TRAINED_SOVITS, flush=True)

    if mode in ("all", "zeroshot"):
        print("=== 零样本（未微调，官方预训练） ===", flush=True)
        tts = TTS(build(PRETRAINED_SOVITS))
        synth(tts, TEXT, PROMPT, ptext, os.path.join(OUT, "gs_zeroshot.wav"))
        del tts
        import torch
        torch.cuda.empty_cache()

    if mode in ("all", "finetune") and TRAINED_SOVITS:
        print("=== 微调后 (SoVITS v2ProPlus) ===", flush=True)
        tts = TTS(build(os.path.join("SoVITS_weights_v2ProPlus", TRAINED_SOVITS)))
        synth(tts, TEXT, PROMPT, ptext, os.path.join(OUT, "gs_finetune.wav"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
