#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LoRA A/B 验收：base vs adapter，同一批验证文案各自生成。

指标（都要看，单独一个都不够）：
  1. spk_cos：生成音频与该说话人真实原声的说话人嵌入余弦
     （用模型自带 speaker_encoder，与部署同源）→ 音色像不像
  2. CER：faster-whisper(CPU int8) 回检，language='ko' → 发音准不准
     🔴 空转写直接判失败，不当作 skipped 放行
  3. 时长：与真实原声（同文本）对比 → 语速/停止行为学没学到
     base 若生成出 2 倍时长 + CER>1，就是"不停嘴/重复跑飞"的铁证

为什么支持多条样本
------------------
单条样本、单个种子不能下结论（历史：单次波动 1~2 分，Qwen 同案例不同 seed
差约 0.95）。`--val-idx 0,1,2,3,4` 一次跑多条，**同一进程内先跑完所有 base
再注入 adapter 跑所有 lora**（顺序不能反：注入后 base 就不纯了）。

用法：
  TTS_LORA_ROOT=/mnt/tts/diag/lora_aligned venv/bin/python tools/lora_eval.py \
      --speaker yohan_aligned \
      --ckpt /mnt/tts/diag/lora_aligned/yohan_aligned/align_v1/ckpt/final \
      --val-idx 0,1,2,3,4
"""
import argparse
import json
import os

import numpy as np
import torch
import librosa
import soundfile as sf

from qwen_tts import Qwen3TTSModel

MODEL_DIR = "/mnt/tts/models/Qwen3-TTS-12Hz-1.7B-Base"
LORA_ROOT = os.environ.get("TTS_LORA_ROOT", "/mnt/tts/data/lora")


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


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", default="yohan")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--val-idx", default="0",
                    help="逗号分隔的验证样本下标，如 0,1,2,3,4")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=os.environ.get(
        "TTS_LORA_AB_OUT", "/mnt/tts/diag/lora_ab"))
    ap.add_argument("--skip-base", action="store_true",
                    help="跳过 base 生成（base 结果已在同 seed 下测过，"
                         "只需补 adapter 时用它省时间）")
    ap.add_argument("--max-new-tokens", type=int, default=800,
                    help="生成上限（12Hz ≈ ×1/12 秒）。🔴 LoRA 早期检查点"
                         "可能退化 EOS 行为不停嘴，必须设上限防跑飞。")
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(LORA_ROOT, args.speaker,
                                           "manifest.json"), encoding="utf-8"))
    ref = manifest["ref"]
    idxs = [int(x) for x in str(args.val_idx).split(",") if x.strip() != ""]
    items = [(i, manifest["val"][i]) for i in idxs if i < len(manifest["val"])]
    if not items:
        raise SystemExit("没有有效验证样本")
    os.makedirs(args.out, exist_ok=True)
    print(f"验证样本 {len(items)} 条：{[i for i, _ in items]}", flush=True)

    print("== 加载模型 ==", flush=True)
    model = Qwen3TTSModel.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    m = model.model

    def gen(text):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        wavs, sr = model.generate_voice_clone(
            text=text, language="Korean",
            ref_audio=ref["wav"], ref_text=ref["ref_text"],
            non_streaming_mode=True,
            max_new_tokens=args.max_new_tokens)
        return wavs[0], sr

    # ---------- 先跑完所有 base ----------
    paths = {}
    if args.skip_base:
        print("== 跳过 base（沿用同 seed 下已测过的 base 结果）==", flush=True)
    else:
        print("== 生成 base ==", flush=True)
        for i, it in items:
            w, sr = gen(it["text"])
            p = os.path.join(args.out, f"{args.speaker}_{i}_base.wav")
            sf.write(p, w, sr)
            paths[(i, "base")] = (p, round(len(w) / sr, 1))
            print(f"  base[{i}] {len(w)/sr:.1f}s -> {p}", flush=True)
        print("base 全部落盘（后续失败也保留）", flush=True)

    # ---------- 再注入 adapter 跑 lora ----------
    if args.ckpt:
        from peft import PeftModel
        print(f"== 注入 adapter {args.ckpt} ==", flush=True)
        PeftModel.from_pretrained(m.talker.model, args.ckpt,
                                  is_trainable=False)
        for i, it in items:
            w, sr = gen(it["text"])
            p = os.path.join(args.out, f"{args.speaker}_{i}_lora.wav")
            sf.write(p, w, sr)
            paths[(i, "lora")] = (p, round(len(w) / sr, 1))
            print(f"  lora[{i}] {len(w)/sr:.1f}s -> {p}", flush=True)

    # ---------- 说话人余弦 ----------
    def spk_emb(path):
        y, sr = librosa.load(path, sr=None, mono=True)
        if sr != m.speaker_encoder_sample_rate:
            y = librosa.resample(y=y.astype(np.float32), orig_sr=int(sr),
                                 target_sr=m.speaker_encoder_sample_rate)
        with torch.no_grad():
            return m.extract_speaker_embedding(audio=y,
                                               sr=m.speaker_encoder_sample_rate)

    def asr(path):
        from faster_whisper import WhisperModel
        # 🔴 必须 local_files_only=True：否则联网检查会卡死（踩过的坑）
        wm = WhisperModel(os.environ.get(
            "TTS_ASR_MODEL", "Systran/faster-whisper-small"),
            device="cpu", compute_type="int8", local_files_only=True)
        audio, _ = librosa.load(path, sr=16000, mono=True)
        segs, _ = wm.transcribe(audio.astype(np.float32), language="ko",
                                beam_size=1)
        hyp = "".join(s.text for s in segs)
        if not hyp.strip():
            raise RuntimeError("ASR返回空串，验收失败")
        return hyp

    rows = []
    for i, it in items:
        truth_wav, text = it["wav"], it["text"]
        e_truth = spk_emb(truth_wav)
        rec = dict(idx=i, dur_truth=round(it["dur"], 1))
        for tag in ("base", "lora"):
            if (i, tag) not in paths:
                continue
            p, d = paths[(i, tag)]
            rec[f"dur_{tag}"] = d
            e = spk_emb(p)
            rec[f"spk_cos_{tag}"] = round(torch.nn.functional.
                                          cosine_similarity(
                                              e_truth.float().flatten(),
                                              e.float().flatten(),
                                              dim=0).item(), 4)
            try:
                rec[f"cer_{tag}"] = round(cer(text, asr(p)), 4)
            except Exception as ex:
                rec[f"cer_{tag}"] = f"ASR失败: {ex}"
        rec["text_head"] = text[:40]
        rows.append(rec)
        print(f"  样本{i}: " + "  ".join(
            f"{k}={v}" for k, v in rec.items() if k != "text_head"),
            flush=True)

    print("\n===== 汇总（中位数）=====")
    summ = {}
    for tag in ("base", "lora"):
        cs = [r[f"cer_{tag}"] for r in rows
              if isinstance(r.get(f"cer_{tag}"), float)]
        ds = [r[f"dur_{tag}"] for r in rows if f"dur_{tag}" in r]
        ss = [r[f"spk_cos_{tag}"] for r in rows if f"spk_cos_{tag}" in r]
        tr = [r["dur_truth"] for r in rows if f"dur_{tag}" in r]
        if cs:
            summ[f"cer_{tag}_med"] = round(med(cs), 4)
            summ[f"cer_{tag}_max"] = round(max(cs), 4)
        if ds:
            summ[f"dur_{tag}_med"] = round(med(ds), 1)
            summ[f"dur_truth_med"] = round(med(tr), 1)
            # 时长比：>1.5 视为明显偏长（不停嘴/重复的强信号）
            summ[f"dur_ratio_{tag}_med"] = round(
                med([d / t for d, t in zip(ds, tr) if t > 0]), 2)
        if ss:
            summ[f"spk_cos_{tag}_med"] = round(med(ss), 4)
    for k, v in summ.items():
        print(f"  {k}: {v}")
    print("\n判读：dur_ratio 明显 >1.5 且 CER 高 = 不停嘴/重复；"
          "CER 需与案例原声噪声底 0.2227 比，不是绝对阈值。")

    json.dump(dict(rows=rows, summary=summ),
              open(os.path.join(args.out, f"{args.speaker}_ab.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
