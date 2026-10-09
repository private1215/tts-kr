#!/usr/bin/env python3
"""LoRA A/B 验收：base vs adapter 生成同一段验证文案。

指标（都要看，单独一个都不够）：
  1. spk_cos：生成音频与该说话人真实原声的说话人嵌入余弦
     （用模型自带 speaker_encoder，与部署同源）→ 音色像不像
  2. CER：faster-whisper(CPU int8) 回检，language='ko' → 发音准不准
  3. 时长：与真实原声同文本时长对比 → 语速习惯学没学到

用法：
  venv/bin/python tools/lora_eval.py --speaker yohan \
      --ckpt /mnt/tts/data/lora/yohan/ckpt/best --val-idx 0
"""
import argparse
import json
import os

import numpy as np
import torch
import librosa

from qwen_tts import Qwen3TTSModel

MODEL_DIR = "/mnt/tts/models/Qwen3-TTS-12Hz-1.7B-Base"
LORA_ROOT = "/mnt/tts/data/lora"


def cer(ref: str, hyp: str) -> float:
    """字符级编辑距离 / ref 长度。"""
    r = [c for c in ref if not c.isspace()]
    h = [c for c in hyp if not c.isspace()]
    if not r:
        return 0.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", default="yohan")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--val-idx", type=int, default=0)
    ap.add_argument("--out", default="/mnt/tts/data/lora/ab")
    ap.add_argument("--max-new-tokens", type=int, default=800,
                    help="生成上限（12Hz ≈ ×1/12 秒）。🔴 LoRA 早期检查点"
                         "可能退化 EOS 行为不停嘴，必须设上限防跑飞。")
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(LORA_ROOT, args.speaker,
                                           "manifest.json"), encoding="utf-8"))
    item = manifest["val"][args.val_idx]
    ref = manifest["ref"]
    text, truth_wav = item["text"], item["wav"]
    os.makedirs(args.out, exist_ok=True)

    print("== 加载模型 ==", flush=True)
    model = Qwen3TTSModel.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    m = model.model

    def gen():
        torch.manual_seed(2026)
        np.random.seed(2026)
        wavs, sr = model.generate_voice_clone(
            text=text, language="Korean",
            ref_audio=ref["wav"], ref_text=ref["ref_text"],
            non_streaming_mode=True,
            max_new_tokens=args.max_new_tokens)
        return wavs[0], sr

    print("== 生成 base ==", flush=True)
    wav_base, sr1 = gen()
    import soundfile as sf
    sf.write(os.path.join(args.out, f"{args.speaker}_base.wav"), wav_base, sr1)
    print("base 立即落盘，后续评测失败也保留", flush=True)

    if args.ckpt:
        from peft import PeftModel
        print(f"== 注入 adapter {args.ckpt} ==", flush=True)
        PeftModel.from_pretrained(m.talker.model, args.ckpt,
                                  is_trainable=False)
        wav_lora, sr2 = gen()
    else:
        wav_lora = None

    # 落盘
    import soundfile as sf
    pb = os.path.join(args.out, f"{args.speaker}_base.wav")
    sf.write(pb, wav_base, sr1)
    print(f"base 已存 {pb} ({len(wav_base)/sr1:.1f}s)")
    pl = None
    if wav_lora is not None:
        pl = os.path.join(args.out, f"{args.speaker}_lora.wav")
        sf.write(pl, wav_lora, sr2)
        print(f"lora 已存 {pl} ({len(wav_lora)/sr2:.1f}s)")

    # ---------- 说话人余弦 ----------
    def spk_emb(path):
        y, sr = librosa.load(path, sr=None, mono=True)
        if sr != m.speaker_encoder_sample_rate:
            y = librosa.resample(y=y.astype(np.float32), orig_sr=int(sr),
                                 target_sr=m.speaker_encoder_sample_rate)
        with torch.no_grad():
            return m.extract_speaker_embedding(audio=y,
                                               sr=m.speaker_encoder_sample_rate)

    e_truth = spk_emb(truth_wav)
    res = {}
    for tag, path in (("base", pb), ("lora", pl)):
        if path is None:
            continue
        e = spk_emb(path)
        cos = torch.nn.functional.cosine_similarity(
            e_truth.float().flatten(), e.float().flatten(), dim=0).item()
        res[f"spk_cos_{tag}"] = round(cos, 4)

    # ---------- CER ----------
    def asr(path):
        from faster_whisper import WhisperModel
        wm = WhisperModel("small", device="cpu", compute_type="int8")
        audio, _ = librosa.load(path, sr=16000, mono=True)
        segs, _ = wm.transcribe(audio.astype(np.float32), language="ko", beam_size=1)
        hyp = "".join(s.text for s in segs)
        if not hyp.strip():
            raise RuntimeError("ASR返回空串，验收失败")
        return hyp

    for tag, path in (("base", pb), ("lora", pl)):
        if path is None:
            continue
        try:
            hyp = asr(path)
            res[f"cer_{tag}"] = round(cer(text, hyp), 4)
        except Exception as e:
            res[f"cer_{tag}"] = f"ASR失败: {e}"

    res["dur_truth"] = round(item["dur"], 1)
    if wav_base is not None:
        res["dur_base"] = round(len(wav_base) / sr1, 1)
    if wav_lora is not None:
        res["dur_lora"] = round(len(wav_lora) / sr2, 1)
    res["val_text_head"] = text[:60]

    print("\n===== A/B 结果 =====")
    for k, v in res.items():
        print(f"  {k}: {v}")
    json.dump(res, open(os.path.join(args.out, f"{args.speaker}_ab.json"),
                        "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
