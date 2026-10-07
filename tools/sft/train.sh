#!/bin/bash
# CosyVoice3 SFT 训练（单卡 3090，锁 UUID，限功耗）
#
# 用法： bash tools/sft/train.sh <数据目录> [conf] [exp名字]
#   bash tools/sft/train.sh /mnt/tts/data/sft/male conf/sweep_1e-5.yaml sweep_1e-5
#
# 🔴 三条硬约束（2026-10-07 掉卡事故后）：
#   1. 3090 必须限 250W（开机自启服务已做，这里再确认一次）
#   2. 训练时不要同时跑 cosy 在线推理（叠加电流）
#   3. 一切产物/日志写 /mnt/tts/**，不写 /tmp（重启即清空）
set -u
DATA=${1:-/mnt/tts/data/sft/male}
CONF=${2:-conf/sweep_1e-5.yaml}
EXP=${3:-sweep_1e-5}

COSY=/mnt/tts/projects/cosy
REPO=$COSY/repo
EXDIR=$REPO/examples/ko_male/cosyvoice3
VENVBIN=$COSY/venv/bin
GPU_UUID=GPU-b67eee2c-c672-6915-910d-617c62abebee   # 3090（nvidia-smi 编号 1）

echo "== 0/3 硬件与功耗 =="
nvidia-smi -i 1 --query-gpu=name,power.limit,temperature.gpu --format=csv
PL=$(nvidia-smi -i 1 --query-gpu=power.limit --format=csv,noheader,nounits | tr -d ' .')
if [ "${PL%%.*}" -gt 250 ]; then
  echo "  ⚠️ 功耗上限 ${PL}W > 250W，强制下调"
  sudo nvidia-smi -i 1 -pl 250
fi
# cosy 在线推理会抢 3090，训练前停掉（训练完手动 systemctl start cosy-tts）
if systemctl is-active --quiet cosy-tts; then
  echo "  停 cosy-tts（避免与训练抢 3090）"
  sudo systemctl stop cosy-tts
fi

echo "== 1/3 环境 =="
export CUDA_VISIBLE_DEVICES=$GPU_UUID
export PYTHONPATH="$COSY/sft_stub:$REPO:$REPO/third_party/Matcha-TTS:${PYTHONPATH:-}"
# 单卡不要 torchrun（实测 c10d_rendezvous_backend 段错误），直接跑 train.py
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29511 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0
mkdir -p /mnt/tts/diag

echo "== 2/3 启动训练（日志：/mnt/tts/diag/sft_${EXP}.log）=="
cd "$EXDIR" || exit 1
setsid nohup "$VENVBIN/python" train.py \
    --config "$CONF" \
    --train_data "$DATA/train/data.list" \
    --cv_data "$DATA/dev/data.list" \
    --model_dir "$EXDIR" \
    --tensorboard_dir "$EXDIR/tensorboard/$EXP" \
    --checkpoint "$EXDIR/exp/$EXP/init.pt" \
    --num_workers 2 \
    --prefetch 8 \
    --pin_memory \
    > "/mnt/tts/diag/sft_${EXP}.log" 2>&1 &
echo "  pid $!"
echo "  跟踪： tail -f /mnt/tts/diag/sft_${EXP}.log"
echo "  监控： bash tools/pcie_watch.sh 3600 /mnt/tts/diag/watch_${EXP}.log"

echo "== 3/3 提醒 =="
echo "  · 甜点在 1 个 epoch 内（step 70 CV 最优 2.401；epoch 5 之后灾难性过拟合）"
echo "  · save_per_step 要小，CV 评估跟着 save 走"
echo "  · 训练完： sudo systemctl start cosy-tts 恢复在线服务"
