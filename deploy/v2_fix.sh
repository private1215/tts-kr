#!/bin/bash
# 第十二版：F0 对齐 + 高频回压（-6.5dB）
set -e
PY=/mnt/tts/projects/omnivoice/venv/bin/python
cd /mnt/tts/diag

echo "== x8197 v2 =="
$PY fix_chain.py x8197/vox_male_raw.wav x8197/m_v2.wav \
  --gender m --target-cps 8.37 --chars 4489

echo "== cf9 v2 =="
$PY fix_chain.py cf9/vox_male_raw.wav cf9/m_v2.wav \
  --gender m --target-cps 8.37 --chars 3919
echo ALLDONE
