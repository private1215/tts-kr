# -*- coding: utf-8 -*-
"""GPT-SoVITS 按**说话人组**推理，用于验证微调收益。

🔴 诚实性前提（不能省，否则分数是假的）
----------------------------------------
prompt 与测试文本都必须取自 **hold 案例的前 HOLD_SEC 秒** —— 这段没有进过
训练集（见 gs_prep_group.py）。如果拿训练素材当提示，等于开卷考试，
相似度再高也不能信。

对比口径
--------
  参照 = hold 案例前 HOLD_SEC 秒的原声（ElevenLabs 生成）
  文本 = 这段原声的 ASR 转写（模型没见过的文本）
  · GS 微调   ：用测试区 5~15s 作 prompt
  · GS 零样本 ：同一 prompt，但用官方预训练权重（不微调）
  两者与参照比 campplus 相似度，差值就是微调带来的收益。

用法
----
    python gs_infer_group.py --spk SOLA  --hold 00b6a45ad6aa --epoch 40
    python gs_infer_group.py --spk yohan --hold 1a6c20a60481 --epoch 40
"""
import os
import sys
import glob
import argparse

import numpy as np
import soundfile as sf

REPO = "/mnt/tts/projects/gpt-sovits/repo"
GS = "/mnt/tts/projects/gpt-sovits"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))
os.chdir(REPO)

import yaml            # noqa: E402
import torch           # noqa: E402
import torchaudio      # noqa: E402


# 🔴 本机 torchaudio 走 torchcodec，缺 libav*.so.58 / libnppicc.so.12，
#    一律用 soundfile 绕开（与 gs_infer.py 同一招）
def _sf_load(path):
    w, sr = sf.read(path, dtype="float32")
    t = torch.from_numpy(np.asarray(w))
    if t.ndim == 1:
        t = t.unsqueeze(0)
    else:
        t = t.transpose(0, 1)
    return t, sr


torchaudio.load = _sf_load

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config   # noqa: E402

BASE = os.environ.get("TTS_BASE", "").strip() or "/mnt/tts"
CASES_DIR = os.path.join(BASE, "data", "cases")
OUT = os.path.join(GS, "out")
os.makedirs(OUT, exist_ok=True)

HOLD_SEC = 60.0
GPT_W = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
PRETRAINED_SOVITS = "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth"


def pick_weight(spk, epoch):
    d = os.path.join(REPO, "SoVITS_weights_v2ProPlus")
    pat = "%s_v2pro_e%s_s*.pth" % (spk, epoch)
    fs = sorted(glob.glob(os.path.join(d, pat)))
    if not fs:
        fs = sorted(glob.glob(os.path.join(d, "%s_v2pro_e*.pth" % spk)))
    if not fs:
        raise SystemExit("找不到 %s 的权重（目录 %s）" % (spk, d))
    return os.path.relpath(fs[-1], REPO)


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
        "use_cuda_graph": False,     # 本机 cuda_graph 捕获会卡住
    }):
        out = item
    if out is None:
        raise RuntimeError("推理没有产出音频")
    sr, audio = out
    audio = np.asarray(audio)
    sf.write(path, audio, sr, subtype="PCM_16")
    print("   ->", path, "%.1fs @%dHz" % (len(audio) / sr, sr), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spk", required=True)
    ap.add_argument("--hold", required=True, help="留出作测试区的案例 id")
    ap.add_argument("--epoch", default="40")
    ap.add_argument("--skip-zeroshot", action="store_true",
                    help="只跑微调版（省一半时间）")
    args = ap.parse_args()

    merged = os.path.join(CASES_DIR, args.hold, "merged.wav")
    if not os.path.exists(merged):
        raise SystemExit("找不到 %s" % merged)

    tag = "%s_%s" % (args.spk.lower(), args.hold[:8])
    ref_wav = os.path.join(OUT, "ref_%s.wav" % tag)      # 参照原声
    prompt = os.path.join(OUT, "prompt_%s.wav" % tag)    # GS 用的 prompt
    txt_path = os.path.join(OUT, "prompt_%s.txt" % tag)

    x, sr = sf.read(merged, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    hold_n = int(HOLD_SEC * sr)
    ref = x[:hold_n]
    sf.write(ref_wav, ref, sr, subtype="PCM_16")
    print("参照（测试区原声 %.0fs）-> %s" % (HOLD_SEC, ref_wav), flush=True)

    # prompt 取测试区内 5~15s，保证没被训练过
    seg = x[int(5 * sr):int(15 * sr)]
    sf.write(prompt, seg, sr, subtype="PCM_16")

    # 生成文本 = 测试区 60s 的完整转写（prompt_text 单独用 5~15s 那段的）。
    # 用整段而不是 10 秒：10 秒只有约 60 字，样本太短、相似度不稳定。
    text_path = os.path.join(OUT, "text_%s.txt" % tag)
    if os.path.exists(text_path) and os.path.exists(txt_path):
        ptext = open(text_path, encoding="utf-8").read().strip()
        ptext_prompt = open(txt_path, encoding="utf-8").read().strip()
    else:
        os.environ.setdefault("HF_HOME", "/mnt/tts/models/asr")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from faster_whisper import WhisperModel
        m = WhisperModel("medium", device="cuda", compute_type="int8")
        segs, _ = m.transcribe(ref_wav, language="ko", beam_size=1)
        ptext = " ".join(s.text for s in segs).strip()
        segs2, _ = m.transcribe(prompt, language="ko", beam_size=1)
        ptext_prompt = " ".join(s.text for s in segs2).strip()
        open(text_path, "w", encoding="utf-8").write(ptext)
        open(txt_path, "w", encoding="utf-8").write(ptext_prompt)
    print("生成文本(%d字): %s" % (len(ptext), ptext[:50] + "..." if len(ptext) > 50 else ptext),
          flush=True)
    print("prompt 文本(%d字): %s" % (len(ptext_prompt), ptext_prompt[:40]), flush=True)

    w = pick_weight(args.spk, args.epoch)
    print("微调权重:", w, flush=True)

    if not args.skip_zeroshot:
        print("=== 零样本（官方预训练，未微调）===", flush=True)
        tts = TTS(build(PRETRAINED_SOVITS))
        synth(tts, ptext, prompt, ptext_prompt,
              os.path.join(OUT, "gs_%s_zeroshot.wav" % tag))
        del tts
        torch.cuda.empty_cache()

    print("=== 微调后 ===", flush=True)
    tts = TTS(build(w))
    synth(tts, ptext, prompt, ptext_prompt,
          os.path.join(OUT, "gs_%s_ft_e%s.wav" % (tag, args.epoch)))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
