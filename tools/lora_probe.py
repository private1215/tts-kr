#!/usr/bin/env python3
"""Qwen3-TTS LoRA 可训性探针（1 条样本跑通 fwd+bwd）。

目的：用 30 分钟代价换三个答案——
  1. teacher-forced 训练路径能不能跑通（官方 forward_sub_talker_finetune
     + 自建主干 codec loss）
  2. loss 数值是否 sane（有限、量级合理）
  3. 3090 上 LoRA fwd+bwd 的峰值显存

样本：yohan_male.wav（15 s 克隆源）「自复制」——ref 与 target 是同一段，
模型在上下文里见过这段 codec，loss 应显著低于随机基线；这也是后续
「过拟合 sanity check」的基础。

关键实现（全部复刻官方 generate() 的 ICL 分支，勿自创格式）：
  - 文本 prompt: <|im_start|>assistant\\n{text}<|im_end|>\\n<|im_start|>assistant\\n
  - ref prompt:   <|im_start|>assistant\\n{ref_text}<|im_end|>\\n
  - prompt embeds = role(3 tok) + codec_prefill + icl_embed(non_streaming)
  - 每步 target 输入 = 16 组 codec 嵌入求和 + tts_pad（talker.forward 生成分支）
  - 主 token loss: h[P-1+t] -> codec_head -> codes[t,0]（对齐生成采样）
  - 子说话人 loss: forward_sub_talker_finetune（官方微调入口，内部自带 shift）

用法（服务器 3090）：
  cd /mnt/tts/projects/qwen3-tts
  CUDA_VISIBLE_DEVICES=GPU-b67eee2c-... venv/bin/python tools/lora_probe.py
"""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
import librosa

MODEL_DIR = "/mnt/tts/models/Qwen3-TTS-12Hz-1.7B-Base"
VOICE_WAV = "/mnt/tts/data/voices/yohan_male.wav"
VOICES_JSON = "/mnt/tts/data/voices/voices.json"
LANG = "korean"
LORA_R = 16

from qwen_tts import Qwen3TTSModel  # noqa: E402

dev = "cuda"
print("== 加载模型 ==", flush=True)
model = Qwen3TTSModel.from_pretrained(
    MODEL_DIR, torch_dtype=torch.bfloat16, device_map=dev,
    attn_implementation="sdpa")
m = model.model                      # Qwen3TTSForConditionalGeneration
talker = m.talker
cfg = m.config.talker_config

for p in m.parameters():
    p.requires_grad_(False)
m.eval()

# ---------- 数据：ref = target = yohan_male 15s ----------
voices = json.load(open(VOICES_JSON, encoding="utf-8"))
ref_text = (voices["yohan_male"]["ref_text"] or "").strip()
wav, sr = librosa.load(VOICE_WAV, sr=None)
print(f"ref: {len(wav)/sr:.2f}s @{sr}Hz, text={len(ref_text)}字", flush=True)

with torch.no_grad():
    enc = m.speech_tokenizer.encode([wav], sr=sr)
    codes = enc.audio_codes[0].to(dev).long()          # (T, 16)
    T = codes.shape[0]
    if sr != m.speaker_encoder_sample_rate:
        wav_spk = librosa.resample(y=wav.astype(np.float32),
                                   orig_sr=int(sr),
                                   target_sr=m.speaker_encoder_sample_rate)
    else:
        wav_spk = wav
    spk = m.extract_speaker_embedding(audio=wav_spk,
                                      sr=m.speaker_encoder_sample_rate)

    def tok(s):
        out = model.processor(text=s, return_tensors="pt", padding=True)
        ids = out["input_ids"].to(dev)
        return ids.unsqueeze(0) if ids.dim() == 1 else ids  # [1, T]

    input_id = tok(f"<|im_start|>assistant\n{ref_text}<|im_end|>\n<|im_start|>assistant\n")
    ref_id = tok(f"<|im_start|>assistant\n{ref_text}<|im_end|>\n")

print(f"codes: T={T}, 组数={codes.shape[1]}; "
      f"text tok={input_id.shape[1]}, ref tok={ref_id.shape[1]}", flush=True)

# speech_tokenizer.encode 内部走 inference_mode，产物是 inference tensor，
# 直接当 CE 的 target 会在 backward 报 "cannot be saved for backward"。
# clone 出来变成普通 tensor。
codes = codes.clone()

# ---------- prompt embeds（复刻 generate() ICL 分支，常量无梯度） ----------
with torch.no_grad():
    tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
        talker.get_text_embeddings()(
            torch.tensor([[m.config.tts_bos_token_id,
                           m.config.tts_eos_token_id,
                           m.config.tts_pad_token_id]], device=dev))
    ).chunk(3, dim=1)                                   # 各 [1,1,D]

    lang_id = (cfg.codec_language_id or {}).get(LANG)
    if lang_id is not None:
        codec_prefill = [[cfg.codec_think_id, cfg.codec_think_bos_id,
                          lang_id, cfg.codec_think_eos_id]]
    else:
        codec_prefill = [[cfg.codec_nothink_id, cfg.codec_think_bos_id,
                          cfg.codec_think_eos_id]]
    ce0 = talker.get_input_embeddings()(torch.tensor(codec_prefill, device=dev))
    ce1 = talker.get_input_embeddings()(
        torch.tensor([[cfg.codec_pad_id, cfg.codec_bos_id]], device=dev))
    codec_input_embed = torch.cat(
        [ce0, spk.to(dev, dtype=talker.dtype).view(1, 1, -1), ce1], dim=1)

    role = talker.text_projection(talker.get_text_embeddings()(input_id[:, :3]))
    pre = torch.cat((tts_pad_embed.expand(-1, codec_input_embed.shape[1] - 2, -1),
                     tts_bos_embed), dim=1) + codec_input_embed[:, :-1]
    talker_input_embed = torch.cat((role, pre), dim=1)  # [1, p, D]

    icl_embed, _ = m.generate_icl_prompt(
        text_id=input_id[:, 3:-5], ref_id=ref_id[:, 3:-2], ref_code=codes,
        tts_pad_embed=tts_pad_embed, tts_eos_embed=tts_eos_embed,
        non_streaming_mode=True)
    prompt_embeds = torch.cat([talker_input_embed, icl_embed], dim=1)  # [1,P,D]

    # 每步 target 输入 = 16 组嵌入求和 + pad（talker.forward 生成分支同款）
    ge = [talker.get_input_embeddings()(codes[:, :1])]
    for i in range(1, cfg.num_code_groups):
        ge.append(talker.code_predictor.get_input_embeddings()[i - 1](codes[:, i:i + 1]))
    target_embeds = torch.cat(ge, dim=1).sum(1).unsqueeze(0) + tts_pad_embed  # [1,T,D]

full = torch.cat([prompt_embeds, target_embeds], dim=1)      # [1, P+T, D]
L, P = full.shape[1], prompt_embeds.shape[1]
print(f"序列: P={P}, T={T}, L={L}", flush=True)

attn = torch.ones(1, L, dtype=torch.long, device=dev)
pos, _ = talker.get_rope_index(attn)
if pos.dim() == 2:
    pos = pos.unsqueeze(0).expand(3, -1, -1)
print(f"position_ids: {tuple(pos.shape)}", flush=True)

# ---------- LoRA ----------
from peft import LoraConfig, get_peft_model  # noqa: E402

lcfg = LoraConfig(r=LORA_R, lora_alpha=LORA_R * 2, lora_dropout=0.0,
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                  bias="none")
# inject_adapter 会原地替换 talker.model 树里的 Linear，talker.model(...) 直接生效
_peft = get_peft_model(talker.model, lcfg)
_peft.print_trainable_parameters()

# ---------- fwd + loss ----------
torch.cuda.reset_peak_memory_stats()
out = talker.model(input_ids=None, attention_mask=attn, position_ids=pos,
                   inputs_embeds=full, use_cache=False)
h = out.last_hidden_state                                   # [1, L, D]

logits_main = talker.codec_head(h[:, P - 1:P - 1 + T])      # [1, T, V]
loss_main = F.cross_entropy(logits_main.squeeze(0).float(), codes[:, 0])

hs = h[:, P - 1:P - 1 + T].squeeze(0)                       # [T, D]
sub_logits, loss_sub = talker.forward_sub_talker_finetune(codes, hs)

# 🔴 官方入口的 off-by-one：ForCausalLMLoss 默认再 shift 一格，而 labels
# (codes[:,1:]) 本身已按位置对齐（pos k 的 head 只见过 g0..gk，本该预测
# g_{k+1}）。叠加后变成"隔着一代预测 g_{k+2}"，loss 反而高于随机基线。
# 手动对齐版（训练真正该用的口径）：
sub_logits_f = sub_logits.float()                           # (T, 15, V)
loss_sub_aligned = F.cross_entropy(
    sub_logits_f.reshape(-1, sub_logits_f.shape[-1]),
    codes[:, 1:].reshape(-1))

loss = loss_main + loss_sub_aligned
print(f"\nloss_main={loss_main.item():.4f}  "
      f"loss_sub(官方shift)={loss_sub.item():.4f}  "
      f"loss_sub(对齐)={loss_sub_aligned.item():.4f}  "
      f"total={loss.item():.4f}", flush=True)

# 随机基线参照：log(vocab)
print(f"随机基线 CE ≈ {torch.log(torch.tensor(float(cfg.vocab_size))).item():.2f} "
      f"(vocab={cfg.vocab_size})", flush=True)

# ---------- bwd ----------
loss.backward()

gn, n_grad = [], 0
for n_, p_ in m.named_parameters():
    if p_.requires_grad:
        n_grad += p_.numel()
        if p_.grad is not None and "lora_" in n_:
            gn.append((n_, p_.grad.norm().item()))
print(f"\n可训练参数: {n_grad/1e6:.2f}M; 收到梯度的 LoRA 权重: {len(gn)}")
for n_, g in gn[:6]:
    print(f"  {n_}: grad_norm={g:.4e}")

peak = torch.cuda.max_memory_allocated() / 2**30
print(f"\n峰值显存: {peak:.2f} GiB (3090 总 24 GiB)")
print("✅ 探针全部通过" if gn and loss.isfinite() else "❌ 有问题，看上面输出")
