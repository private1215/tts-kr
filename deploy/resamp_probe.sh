#!/bin/bash
# 变调重采样精度实验：asetrate+aresample 在 16k 上重采样会把 flat 压掉
# raw 0.0369 -> 清洗EQ 0.0292 -> 变调后 0.0236（EL 0.0396）
# 试 soxr 高精度重采样能不能保住
set -e
cd /mnt/tts/diag/cf9
SR=16000
R=1.1785
NEW=$(python3 -c "print(int(round($SR*$R)))")

ffmpeg -y -v error -i vox_male_fix.wav -t 45 \
  -af "asetrate=$NEW,aresample=$SR:resampler=soxr:precision=28,atempo=$(python3 -c "print(1/$R)")" sx_$R.wav
ffmpeg -y -v error -i vox_male_fix.wav -t 45 \
  -af "asetrate=$NEW,aresample=$SR,atempo=$(python3 -c "print(1/$R)")" df_$R.wav
echo built
ls -la sx_$R.wav df_$R.wav
