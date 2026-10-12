#!/usr/bin/env bash
# GPT-SoVITS v2ProPlus：按**说话人组**跑特征提取 / 微调（参数化版）
#
# 为什么要这个：原来的 gs_preprocess.sh / gs_train_s2.sh 把 exp_name、数据集路径
# 全部硬编码成 hh_v2pro（韩华案例）。而案例库实际是 SOLA 22 条 / yohan 13 条，
# 要按人来微调就不能写死。这个脚本接受说话人名，两组互不干扰。
#
# 用法：
#   ./gs_run_group.sh SOLA  preprocess     # 特征提取（HuBERT / 音素 / sv）
#   ./gs_run_group.sh SOLA  train          # 微调 s2（决定音色的声学模型）
#   ./gs_run_group.sh yohan preprocess
#   ./gs_run_group.sh yohan train
#
# 前置：先用 tools/gs_prep_group.py 生成 dataset_<spk>/{train,val}.list
set -e

SPK="${1:?用法: gs_run_group.sh <说话人> <preprocess|train>}"
STEP="${2:?用法: gs_run_group.sh <说话人> <preprocess|train>}"

GS=/mnt/tts/projects/gpt-sovits
REPO=$GS/repo
VENV=$GS/venv/bin/python
DATA=$GS/dataset_${SPK}
EXP=$GS/exp/${SPK}_v2pro
VERSION=v2ProPlus

if [ ! -f "$DATA/train.list" ]; then
  echo "❌ 找不到 $DATA/train.list，先跑 gs_prep_group.py --spk $SPK"
  exit 1
fi
echo "说话人=$SPK  数据集=$DATA  实验=$EXP"

cd "$REPO"
export PYTHONPATH="$REPO:$REPO/GPT_SoVITS"
export PYTHONDONTWRITEBYTECODE=1
# 3090：与主服务 Qwen（3060）隔离，别抢卡
export _CUDA_VISIBLE_DEVICES=GPU-b67eee2c-c672-6915-910d-617c62abebee
export CUDA_VISIBLE_DEVICES=GPU-b67eee2c-c672-6915-910d-617c62abebee
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

PM="$REPO/GPT_SoVITS/pretrained_models"

if [ "$STEP" = "preprocess" ]; then
  export inp_text="$DATA/train.list"
  export inp_wav_dir="$DATA/train"
  export exp_name=${SPK}_v2pro
  export i_part=0
  export all_parts=1
  export opt_dir="$EXP"
  export bert_pretrained_dir="$PM/chinese-roberta-wwm-ext-large"
  export cnhubert_base_dir="$PM/chinese-hubert-base"
  export sv_path="$PM/sv/pretrained_eres2netv2w24s4ep4.ckpt"
  export version=$VERSION
  export is_half=True

  mkdir -p "$EXP"
  echo "########## 1-get-text ##########"
  $VENV -s GPT_SoVITS/prepare_datasets/1-get-text.py
  echo "########## 2-get-hubert-wav32k ##########"
  $VENV -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py
  echo "########## 2-get-sv（v2Pro 系列需要）##########"
  $VENV -s GPT_SoVITS/prepare_datasets/2-get-sv.py
  echo "产出：2-name2text=$(wc -l < "$EXP/2-name2text-0.txt" 2>/dev/null) " \
       "4-cnhubert=$(ls "$EXP/4-cnhubert" 2>/dev/null | wc -l) " \
       "5-wav32k=$(ls "$EXP/5-wav32k" 2>/dev/null | wc -l) " \
       "7-sv_cn=$(ls "$EXP/7-sv_cn" 2>/dev/null | wc -l)"
  echo FEAT_DONE
  exit 0
fi

if [ "$STEP" = "train" ]; then
  # WebUI 会合并多分片，headless 跑要自己做
  if [ ! -f "$EXP/2-name2text.txt" ]; then
    cat "$EXP"/2-name2text-*.txt > "$EXP/2-name2text.txt"
  fi
  echo "2-name2text.txt 行数: $(wc -l < "$EXP/2-name2text.txt")"
  # 🔴 这两个目录 WebUI 在开训前 makedirs，headless 必须自己建，
  #    否则 save_every_epoch 保存时直接 FileNotFoundError 崩掉
  mkdir -p "$EXP/logs_s2_${VERSION}" "$REPO/SoVITS_weights_${VERSION}"

  $VENV - <<PY
import json
cfg = "GPT_SoVITS/configs/s2v2ProPlus.json"
d = json.load(open(cfg))
exp = "$EXP"
d["train"]["batch_size"] = 8
d["train"]["epochs"] = 40
d["train"]["text_low_lr_rate"] = 0.4
d["train"]["pretrained_s2G"] = "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth"
d["train"]["pretrained_s2D"] = "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2ProPlus.pth"
d["train"]["if_save_latest"] = True
d["train"]["if_save_every_weights"] = True
d["train"]["save_every_epoch"] = 10
d["train"]["gpu_numbers"] = "0"
d["train"]["grad_ckpt"] = False
d["train"]["lora_rank"] = None
d["model"]["version"] = "$VERSION"
d["data"]["exp_dir"] = d["s2_ckpt_dir"] = exp
d["save_weight_dir"] = "SoVITS_weights_${VERSION}"
d["name"] = "${SPK}_v2pro"
d["version"] = "$VERSION"
json.dump(d, open("$GS/tmp_s2_${SPK}.json", "w"))
print("配置已写入 $GS/tmp_s2_${SPK}.json")
PY

  echo "########## 训练 SoVITS ${VERSION}（说话人=$SPK）##########"
  $VENV -s GPT_SoVITS/s2_train.py --config "$GS/tmp_s2_${SPK}.json"
  echo TRAIN_DONE
  ls -la "$REPO/SoVITS_weights_${VERSION}/" 2>/dev/null | tail -12
  exit 0
fi

echo "❌ 未知步骤: $STEP（只能是 preprocess 或 train）"
exit 1
