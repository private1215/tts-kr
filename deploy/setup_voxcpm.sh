#!/bin/bash
# VoxCPM2 隔离安装（v2：源码已 rsync，跳过 git clone）—— 2026-10-13
set -x
cd /mnt/tts/projects/voxcpm || exit 1

if [ ! -x venv/bin/python ]; then
  python3.12 -m venv venv || exit 3
fi
venv/bin/pip install -q --upgrade pip

# torch 大包先装
venv/bin/pip install torch torchaudio >> /tmp/voxcpm_pip.log 2>&1 || exit 4
# 项目其余依赖
cd repo && ../venv/bin/pip install -e . >> /tmp/voxcpm_pip.log 2>&1 || exit 5
cd ..

# 权重走 hf-mirror
export HF_ENDPOINT=https://hf-mirror.com
venv/bin/python - <<'PY' >> /tmp/voxcpm_pip.log 2>&1 || exit 6
from huggingface_hub import snapshot_download
p = snapshot_download("openbmb/VoxCPM2", local_dir="/mnt/tts/projects/voxcpm/models/VoxCPM2")
print("WEIGHTS_DONE", p)
PY

echo "ALL_DONE"
