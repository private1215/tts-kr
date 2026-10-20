#!/bin/bash
# make_0902_clips.sh —— 给员工反馈的 16 个时间点切片 + 画声谱图
# 用法: bash make_0902_clips.sh
set -u
MP3=/mnt/tts/output/202610190902/202610190902-1.mp3
OUT=/mnt/tts/diag/v0902/clips
mkdir -p "$OUT"

# 时间点:秒|标签
PTS="0|p00_开头怪音 5|p05_磁音 14|p14_少一个字 29|p29_吃字 44|p44_额 58|p58_额 64|p64_兹音 73|p73_吃字 83|p83_未标注 88|p88_未标注 100|p100_啊 115|p115_没音 127|p127_额 135|p135_兹音 142|p142_额"

for item in $PTS; do
  t="${item%%|*}"; lab="${item##*|}"
  start=$(python3 -c "print(max(0.0, float('$t')-2.5))")
  # 片段：前后各 2.5s，共 5s
  ffmpeg -y -v error -ss "$start" -t 5 -i "$MP3" -codec:a libmp3lame -b:a 128k \
      "$OUT/${lab}_${t}s.mp3"
  # 声谱图：该点前后各 3s
  s2=$(python3 -c "print(max(0.0, float('$t')-3.0))")
  ffmpeg -y -v error -ss "$s2" -t 6 -i "$MP3" \
      -filter_complex "showspectrumpg=s=1200x420:mode=combined:slide=scroll:legend=1:orientation=vertical" \
      -frames:v 1 "$OUT/${lab}_${t}s.png"
done
ls -la "$OUT"
echo DONE
