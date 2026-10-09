#!/usr/bin/env python3
"""Qwen3-TTS LoRA 正式训练（单人）。

与探针(tools/lora_probe.py)同一条 teacher-forced 路径：
  - prompt 构造完全复刻官方 generate() 的 ICL 分支(non_streaming)
  - 主 token loss: codec_head(h[P-1+t]) -> codes[t,0]（手动对齐，无 shift）
  - 子说话人 loss: forward_sub_talker_finetune 的 logits 手动对齐 CE
    🔴 不用它返回的 loss（官方 ForCausalLMLoss 自带 off-by-one）
  - ref 固定用克隆源音色（与线上部署同一份参考音频 → 训练即部署）

显存实测 ~6-8 GiB（3090），可与服务共存。
用法（服务器后台）：
  cd /mnt/tts/projects/qwen3-tts
  CUDA_VISIBLE_DEVICES=<3090UUID> venv/bin/python tools/lora_train.py \
      --speaker yohan --epochs 3 --lr 1e-4
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import librosa

from qwen_tts import Qwen3TTSModel
from peft import LoraConfig, get_peft_model

MODEL_DIR = "/mnt/tts/models/Qwen3-TTS-12Hz-1.7B-Base"
LORA_ROOT = os.environ.get("TTS_LORA_ROOT", "/mnt/tts/data/lora")
LANG = "korean"


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", default="yohan")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--run-name", default="text_eos_v2")
    ap.add_argument("--max-steps", type=int, default=20,
                    help="先做有界验证；通过生成验收后再扩大训练")
    return ap.parse_args()


def main():
    args = parse()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = "cuda"

    out_root = os.path.join(LORA_ROOT, args.speaker)
    manifest = json.load(open(os.path.join(out_root, "manifest.json"),
                              encoding="utf-8"))
    run_dir = os.path.join(out_root, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train.log")
    log = open(log_path, "a", encoding="utf-8")

    def P(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    P(f"\n===== LoRA 训练 {args.speaker} lr={args.lr} r={args.r} "
      f"epochs={args.epochs} =====")

    model = Qwen3TTSModel.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map=dev,
        attn_implementation="sdpa")
    m = model.model
    talker = m.talker
    cfg = m.config.talker_config

    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()

    lcfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=0.0,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      bias="none")
    peft_model = get_peft_model(talker.model, lcfg)
    peft_model.print_trainable_parameters()

    # ---------- 固定的 ref prompt（克隆源，与部署一致） ----------
    ref = manifest["ref"]
    ref_text = (ref["ref_text"] or "").strip()
    with torch.no_grad():
        rwav, rsr = librosa.load(ref["wav"], sr=None)
        if rsr != m.speaker_encoder_sample_rate:
            rwav_spk = librosa.resample(y=rwav.astype(np.float32),
                                         orig_sr=int(rsr),
                                         target_sr=m.speaker_encoder_sample_rate)
        else:
            rwav_spk = rwav
        spk = m.extract_speaker_embedding(audio=rwav_spk,
                                          sr=m.speaker_encoder_sample_rate)

        def tok(s):
            out = model.processor(text=s, return_tensors="pt", padding=True)
            ids = out["input_ids"].to(dev)
            return ids.unsqueeze(0) if ids.dim() == 1 else ids

        input_id = tok(f"<|im_start|>assistant\n{ref_text}<|im_end|>\n<|im_start|>assistant\n")
        ref_id = tok(f"<|im_start|>assistant\n{ref_text}<|im_end|>\n")

        tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
            talker.get_text_embeddings()(
                torch.tensor([[m.config.tts_bos_token_id,
                               m.config.tts_eos_token_id,
                               m.config.tts_pad_token_id]], device=dev))
        ).chunk(3, dim=1)

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
        talker_input_embed = torch.cat((role, pre), dim=1)

        # ref_code（克隆源的 codec，进 ICL 上下文）
        renc = m.speech_tokenizer.encode([rwav], sr=rsr)
        ref_codes = renc.audio_codes[0].to(dev).long()

        icl_embed, _ = m.generate_icl_prompt(
            text_id=input_id[:, 3:-5], ref_id=ref_id[:, 3:-2], ref_code=ref_codes,
            tts_pad_embed=tts_pad_embed, tts_eos_embed=tts_eos_embed,
            non_streaming_mode=True)
        prompt_embeds = torch.cat([talker_input_embed, icl_embed], dim=1)
        P_PROMPT = prompt_embeds.shape[1]
        P(f"prompt 长度 P={P_PROMPT} (ref {len(rwav)/rsr:.1f}s)")

    def build_target_embeds(codes):
        """每步 target 输入 = 16 组 codec 嵌入求和 + tts_pad（生成分支同款）。"""
        ge = [talker.get_input_embeddings()(codes[:, :1])]
        for i in range(1, cfg.num_code_groups):
            ge.append(talker.code_predictor.get_input_embeddings()[i - 1](
                codes[:, i:i + 1]))
        return torch.cat(ge, dim=1).sum(1).unsqueeze(0) + tts_pad_embed

    # ---------- 预编码全部样本（一次性，之后纯 GPU） ----------
    def encode_items(items):
        out = []
        for n, it in enumerate(items):
            wav, sr = librosa.load(it["wav"], sr=None)
            enc = m.speech_tokenizer.encode([wav], sr=sr)
            codes = enc.audio_codes[0].to(dev).long().clone()   # 🔴 inference tensor 要 clone
            txt_id = tok(f"<|im_start|>assistant\n{it['text']}<|im_end|>\n<|im_start|>assistant\n")
            out.append((codes, txt_id))
            if (n + 1) % 50 == 0:
                P(f"  预编码 {n+1}/{len(items)}")
        return out

    P(f"预编码 train {len(manifest['train'])} 片 ...")
    t0 = time.time()
    train_set = encode_items(manifest["train"])
    val_set = encode_items(manifest["val"])
    P(f"预编码完成 {time.time()-t0:.0f}s（val {len(val_set)} 片）")

    # ---------- loss ----------
    def compute_loss(codes, txt_id):
        """codes [T,16]; txt_id [1,n]. 返回 (main, sub_aligned, total)。"""
        with torch.no_grad():
            # 每个目标必须以自己的文字作为条件，不能重复参考稿。
            target_icl, _ = m.generate_icl_prompt(
                text_id=txt_id[:, 3:-5], ref_id=ref_id[:, 3:-2],
                ref_code=ref_codes, tts_pad_embed=tts_pad_embed,
                tts_eos_embed=tts_eos_embed, non_streaming_mode=True)
            sample_prompt = torch.cat([talker_input_embed, target_icl], dim=1)
            prompt_len = sample_prompt.shape[1]
            target_embeds = build_target_embeds(codes)
            full = torch.cat([sample_prompt, target_embeds], dim=1)
            L = full.shape[1]
            attn = torch.ones(1, L, dtype=torch.long, device=dev)
            pos, _ = talker.get_rope_index(attn)
            if pos.dim() == 2:
                pos = pos.unsqueeze(0).expand(3, -1, -1)
        # 注意：full 是 no_grad 常量输入；梯度经 LoRA attention 层回传
        full = full.detach().requires_grad_(False)
        out = talker.model(input_ids=None, attention_mask=attn,
                           position_ids=pos, inputs_embeds=full,
                           use_cache=False)
        h = out.last_hidden_state
        T = codes.shape[0]
        # 最后一个音频帧输入之后预测 EOS；EOS 不参与子码本损失。
        logits_main = talker.codec_head(h[:, prompt_len - 1:prompt_len + T])
        labels_main = torch.cat([codes[:, 0], torch.tensor(
            [cfg.codec_eos_token_id], device=dev, dtype=torch.long)])
        loss_main = F.cross_entropy(logits_main.squeeze(0).float(), labels_main)
        hs = h[:, prompt_len - 1:prompt_len - 1 + T].squeeze(0)
        sub_logits, _ = talker.forward_sub_talker_finetune(codes, hs)
        # 🔴 手动对齐 CE（官方返回的 loss 自带 off-by-one，不能用）
        sl = sub_logits.float()
        loss_sub = F.cross_entropy(sl.reshape(-1, sl.shape[-1]),
                                   codes[:, 1:].reshape(-1))
        return loss_main, loss_sub, loss_main + loss_sub

    # ---------- 训练循环 ----------
    trainable = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    step = 0
    ckpt_root = os.path.join(run_dir, "ckpt")
    os.makedirs(ckpt_root, exist_ok=True)

    def val_loss():
        peft_model.eval()
        with torch.no_grad():
            ls = [compute_loss(c, t)[2].item() for c, t in val_set]
        peft_model.train()
        return float(np.mean(ls))

    def save_ckpt(tag):
        d = os.path.join(ckpt_root, tag)
        peft_model.save_pretrained(d)
        P(f"  💾 已存 {d}")

    peft_model.train()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    best_val, best_tag = 1e9, None
    for ep in range(args.epochs):
        order = list(range(len(train_set)))
        random.shuffle(order)
        P(f"\n-- epoch {ep+1}/{args.epochs} ({len(order)} 步) --")
        ep_loss, n = 0.0, 0
        for oi in order:
            codes, txt_id = train_set[oi]
            loss_main, loss_sub, loss = compute_loss(codes, txt_id)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            step += 1
            ep_loss += loss.item(); n += 1
            if step % 10 == 0:
                P(f"  step {step}  loss={loss.item():.3f} "
                  f"(main {loss_main.item():.3f} / sub {loss_sub.item():.3f}) "
                  f"avg={ep_loss/n:.3f}  {time.time()-t0:.0f}s")
            if step % args.val_every == 0:
                vl = val_loss()
                P(f"  📊 step {step} VAL loss={vl:.3f}")
                if vl < best_val:
                    best_val, best_tag = vl, f"step{step}"
                    save_ckpt(f"best")
            if step % args.save_every == 0:
                save_ckpt(f"step{step}")
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    vl = val_loss()
    P(f"\n最终 VAL loss={vl:.3f}（best={best_val:.3f} @ {best_tag}）")
    save_ckpt("final")
    peak = torch.cuda.max_memory_allocated() / 2**30
    P(f"峰值显存 {peak:.2f} GiB；总步数 {step}；总时长 {(time.time()-t0)/60:.1f} 分钟")
    P("===== 训练结束 =====")


if __name__ == "__main__":
    main()
