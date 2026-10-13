#!/bin/bash
# OmniVoice 隔离安装 —— 2026-10-13
# 与 VoxCPM 各自独立 venv，不共享、不碰生产。
set -x
cd /mnt/tts/projects/omnivoice || exit 1

if [ ! -x venv/bin/python ]; then
  python3.12 -m venv venv || exit 3
fi
venv/bin/pip install -q --upgrade pip

# torch 先装（大包）
venv/bin/pip install torch torchaudio >> /tmp/omni_pip.log 2>&1 || exit 4
# 项目依赖（跳过构建隔离的 scm 版本问题：OmniVoice 用静态版本）
cd repo && ../venv/bin/pip install -e . >> /tmp/omni_pip.log 2>&1 || {
  # 回退：不装包体，直接加 PYTHON_PATH 也能跑
  echo "EDITABLE_FAILED，改用 PYTHONPATH 方式" >> /tmp/omni_pip.log
  ../venv/bin/pip install accelerate pydub tensorboardX webdataset numpy soundfile librosa >> /tmp/omni_pip.log 2>&1 || exit 5
}
cd ..

# 权重走 hf-mirror，禁用 xet
export HF_ENDPOINT=https://hf-mirror.com
venv/bin/python - <<'PY' >> /tmp/omni_pip.log 2>&1 || exit 6
from huggingface_hub import snapshot_download
import os
p = snapshot_download("k2-fsa/OmniVoice", local_dir="/mnt/tts/projects/omnivoice/models/OmniVoice",
                      max_workers=4)
print("WEIGHTS_DONE", p)
PY

echo "ALL_DONE"
