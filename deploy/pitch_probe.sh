#!/bin/bash
# 变调实验 v2：pitch 是频率比(ratio)。目标 ratio = 198/167.8 ≈ 1.18
set -e
cd /mnt/tts/diag/cf9
for r in 1.10 1.15 1.18 1.22; do
  ffmpeg -y -v error -i vox_male_fix.wav -t 60 \
    -af "rubberband=pitch=${r}" "r_${r}.wav"
  echo "ok ${r}"
done
ls -la r_*.wav
