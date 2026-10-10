#!/usr/bin/env bash
# 微调 SoVITS v2ProPlus（只训 s2 = 决定音色的声学模型，GPT/s1 用预训练不动）
#
# 为什么只训 s2：我们只有约 6 分钟目标说话人音频。
#   - s2 决定音色，是这次要解决的核心（员工反馈的"不像那个人"）
#   - s1 决定文本→韵律节奏，同时训容易把预训练的语言能力带偏（灾难性遗忘），
#     收益还得靠人耳盲听才知。先做低风险的一步，拿到对照结果再决定要不要加 s1。
set -e

REPO=/mnt/tts/projects/gpt-sovits/repo
VENV=/mnt/tts/projects/gpt-sovits/venv/bin/python
EXP=/mnt/tts/projects/gpt-sovits/exp/hh_v2pro
cd "$REPO"

export PYTHONPATH="$REPO:$REPO/GPT_SoVITS"
export CUDA_VISIBLE_DEVICES=GPU-b67eee2c-c672-6915-910d-617c62abebee   # 3090
export HF_HUB_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com

# WebUI 会把多卡的 -N 分片合并成 2-name2text.txt，headless 跑要自己做这一步
if [ ! -f "$EXP/2-name2text.txt" ]; then
  cat "$EXP"/2-name2text-*.txt > "$EXP/2-name2text.txt"
fi
echo "2-name2text.txt 行数: $(wc -l < "$EXP/2-name2text.txt")"

# 🔴 这两个目录 WebUI 是在开训前 makedirs 的，headless 跑必须自己建，
#    否则第 save_every_epoch 个 epoch 保存时直接 FileNotFoundError 崩掉
mkdir -p "$EXP/logs_s2_v2ProPlus" "$REPO/SoVITS_weights_v2ProPlus"

$VENV - <<'PY'
import json, os
cfg = "GPT_SoVITS/configs/s2v2ProPlus.json"
d = json.load(open(cfg))
exp = "/mnt/tts/projects/gpt-sovits/exp/hh_v2pro"
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
d["model"]["version"] = "v2ProPlus"
d["data"]["exp_dir"] = d["s2_ckpt_dir"] = exp
d["save_weight_dir"] = "SoVITS_weights_v2ProPlus"
d["name"] = "hh_v2pro"
d["version"] = "v2ProPlus"
out = "/mnt/tts/projects/gpt-sovits/tmp_s2.json"
json.dump(d, open(out, "w"))
print("配置已写入", out)
PY

echo "########## 开始训练 SoVITS v2ProPlus ##########"
$VENV -s GPT_SoVITS/s2_train.py --config /mnt/tts/projects/gpt-sovits/tmp_s2.json
echo TRAIN_DONE
ls -la "$REPO/SoVITS_weights_v2ProPlus/" 2>/dev/null | tail -12
