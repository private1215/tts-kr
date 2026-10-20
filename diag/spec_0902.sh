#!/bin/bash
# spec_0902.sh —— 用 showspectrumpic 给 15 个反馈点画声谱图
set -u
MP3=/mnt/tts/output/202610190902/202610190902-1.mp3
OUT=/mnt/tts/diag/v0902/clips
mkdir -p "$OUT"
for t in 0 5 14 29 44 58 64 73 83 88 100 115 127 135 142; do
  s=$(python3 -c "print(max(0.0,$t-3.0))")
  ffmpeg -y -v error -ss "$s" -t 6 -i "$MP3" \
      -filter_complex "showspectrumpic=s=1400x500:legend=1" \
      -frames:v 1 "$OUT/spec_${t}s.png"
done
ls "$OUT"/*.png | wc -l
echo DONE
