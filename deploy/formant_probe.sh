#!/bin/bash
# 变调保共振峰实验：只升基频、不动声道（音色）特性
# 目的：普通变调会把整个频谱搬上去，hf_ratio 从 0.0059 涨到 0.0185（EL 只有 0.0104）
set -e
cd /mnt/tts/diag/cf9
for r in 1.1785 1.2130; do
  ffmpeg -y -v error -i vox_male_fix.wav -t 45 \
    -af "rubberband=pitch=${r}:formant=preserved" "fp_${r}.wav"
  echo "ok ${r}"
done
ls -la fp_*.wav
