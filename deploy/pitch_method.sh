#!/bin/bash
# 变调方法对比：rubberband vs asetrate+aresample
# 关注 flat_mean（频谱平坦度）：EL 0.0424，rubberband 变调后掉到 0.0348（声音变"过净"=发假）
set -e
cd /mnt/tts/diag/cf9
SR=48000
R=1.1785
# 原始采样率确认
ffprobe -v error -show_entries stream=sample_rate -of csv=p=0 vox_male_fix.wav > sr.txt
SR=$(cat sr.txt)
echo "sr=$SR"

ffmpeg -y -v error -i vox_male_fix.wav -t 45 \
  -af "asetrate=$(python3 -c "print($SR*$R)"),aresample=$SR" as_$R.wav
echo "ok asetrate"
ls -la as_$R.wav fp_$R.wav 2>/dev/null
