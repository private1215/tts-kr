#!/bin/bash
# 第十二轮切片：rubberband 版(旧) vs asetrate 版(新) vs EL 标杆
set -e
cd /mnt/tts/diag
OUT=/mnt/tts/data/tone-ab
S=40; T=35

cut() { ffmpeg -y -v error -ss $S -t $T -i "$1" -ar 44100 -b:a 128k /tmp/$2.mp3; \
        echo 'a19910119' | sudo -S cp /tmp/$2.mp3 "$OUT/$2.mp3" >/dev/null 2>&1 || \
        sudo cp /tmp/$2.mp3 "$OUT/$2.mp3"; }

cut x8197/m_f0.wav  k10_old
cut x8197/m_v4.wav  k10_new
cut /mnt/tts/data/cases/8197d9ac8a35/merged.wav k10_el
cut x8197/m_v4.wav  k10_c1
cut x8197/m_v4.wav  k10_c2

echo "=== 时长 ==="
for f in k10_old k10_new k10_el k10_c1 k10_c2; do
  printf '%s ' $f; ffprobe -v error -show_entries format=duration -of csv=p=0 $OUT/$f.mp3
done
echo "=== LUFS ==="
for f in k10_old k10_new k10_el; do
  printf '%s ' $f; ffmpeg -hide_banner -i $OUT/$f.mp3 -af ebur128 -f null - 2>&1 | grep 'I:' | tail -1
done
