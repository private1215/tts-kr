#!/usr/bin/env bash
# GPT-SoVITS v2ProPlus 特征提取 pipeline（headless，不用 WebUI）
#
# 三个脚本都用环境变量传参（官方 webui.py 就是这么 subprocess 调的），
# 必须从仓库根目录运行，因为脚本内部用相对路径 sys.path.append(now_dir)。
set -e

REPO=/mnt/tts/projects/gpt-sovits/repo
VENV=/mnt/tts/projects/gpt-sovits/venv/bin/python
cd "$REPO"

export PYTHONPATH="$REPO:$REPO/GPT_SOITS"
# GPT_SoVITS/prepare_datasets/*.py 里 `from text.cleaner import ...`
# 依赖 GPT_SoVITS 目录本身在 sys.path 上（官方 install.sh 用 .pth 做的，
# 我们 headless 跑就直接挂 PYTHONPATH）
export PYTHONPATH="/mnt/tts/projects/gpt-sovits/repo:/mnt/tts/projects/gpt-sovits/repo/GPT_SoVITS"
export PYTHONDONTWRITEBYTECODE=1

export _CUDA_VISIBLE_DEVICES=GPU-b67eee2c-c672-6915-910d-617c62abebee   # 3090
export CUDA_VISIBLE_DEVICES=GPU-b67eee2c-c672-6915-910d-617c62abebee
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

EXP=/mnt/tts/projects/gpt-sovits/exp/hh_v2pro
DATA=/mnt/tts/projects/gpt-sovits/dataset
PM="$REPO/GPT_SoVITS/pretrained_models"

export inp_text="$DATA/train.list"
export inp_wav_dir="$DATA/train"
export exp_name=hh_v2pro
export i_part=0
export all_parts=1
export opt_dir="$EXP"
export bert_pretrained_dir="$PM/chinese-roberta-wwm-ext-large"
export cnhubert_base_dir="$PM/chinese-hubert-base"
export sv_path="$PM/sv/pretrained_eres2netv2w24s4ep4.ckpt"
export version=v2ProPlus
export is_half=True

mkdir -p "$EXP"

echo "########## STEP 1: 文本 -> 音素 (1-get-text) ##########"
$VENV -s GPT_SoVITS/prepare_datasets/1-get-text.py

echo "########## STEP 2: HuBERT + 32k 重采样 ##########"
$VENV -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py

echo "########## STEP 3: 说话人向量 sv (v2Pro 系列需要) ##########"
$VENV -s GPT_SoVITS/prepare_datasets/2-get-sv.py

echo "########## 产出检查 ##########"
ls -la "$EXP" | head -20
echo "2-name2text 行数: $(wc -l < "$EXP/2-name2text-0.txt" 2>/dev/null)"
echo "4-cnhubert 个数: $(ls "$EXP/4-cnhubert" 2>/dev/null | wc -l)"
echo "5-wav32k  个数: $(ls "$EXP/5-wav32k" 2>/dev/null | wc -l)"
echo "7-sv_cn   个数: $(ls "$EXP/7-sv_cn" 2>/dev/null | wc -l)"
echo FEAT_DONE
