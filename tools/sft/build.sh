#!/bin/bash
# 数据集 -> 特征 -> parquet：调 CosyVoice3 官方脚本（路径坑已在下面修掉）
#
# 用法： bash tools/sft/build.sh /mnt/tts/data/sft/male
#
# 前置：先跑 tools/sft/prep.py 生成 kaldi 四件套 + instruct + text.normalized
set -u
DATA=${1:-/mnt/tts/data/sft/male}
COSY=/mnt/tts/projects/cosy
REPO=$COSY/repo
VENVBIN=$COSY/venv/bin
GPU_UUID=GPU-b67eee2c-c672-6915-910d-617c62abebee
export CUDA_VISIBLE_DEVICES=$GPU_UUID
# deepspeed 装不上（0.19 要 Dynamo/Py3.12；0.14 要 CUDA_HOME），用最小替身
export PYTHONPATH="$COSY/sft_stub:$REPO:$REPO/third_party/Matcha-TTS:${PYTHONPATH:-}"

for split in train dev; do
  D=$DATA/$split
  [ -f "$D/wav.scp" ] || { echo "🔴 缺 $D/wav.scp，先跑 prep.py"; exit 1; }
  echo "===== $split ====="
  cd "$D" || exit 1

  # ⚠️ 官方 run.sh 里写的是 tools/extract_embedding.py（相对 CWD），
  #    而 tools/ 在**仓库根**，直接跑会 file not found。
  echo "-- 说话人嵌入 (campplus)"
  "$VENVBIN/python" ../../../tools/extract_embedding.py --dir . \
      --onnx_path "$REPO"/campplus.onnx 2>&1 | tail -3

  echo "-- 语音 token (16k)"
  "$VENVBIN/python" ../../../tools/extract_speech_token.py --dir . \
      --onnx_path True 2>&1 | tail -3

  # 🔴 CosyVoice3LM.forward 直接取 instruct_token，缺了就 KeyError
  [ -f instruct ] || echo "You are a helpful assistant.<|endofprompt|>" > instruct

  echo "-- parquet + data.list"
  "$VENVBIN/python" ../../../tools/make_parquet.py \
      --src_dir . --des_dir . --num_utts_per_parquet 100 --num_processes 4
  ls -1 ./*.parquet 2>/dev/null | sed "s|^|$(pwd)/|" > data.list
  echo "  parquet 数: $(wc -l < data.list)"
done

echo
echo "✅ 完成。下一步： bash tools/sft/train.sh $DATA"
