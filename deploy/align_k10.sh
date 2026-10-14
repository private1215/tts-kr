#!/bin/bash
# 响度对齐：线性 volume，不用 loudnorm
set -e
OUT=/mnt/tts/data/tone-ab
adj() { ffmpeg -y -v error -i "$OUT/$1.mp3" -af "volume=$2dB" /tmp/b_$1.mp3; \
        echo 'a19910119' | sudo -S cp /tmp/b_$1.mp3 "$OUT/$1.mp3" >/dev/null 2>&1 || \
        sudo cp /tmp/b_$1.mp3 "$OUT/$1.mp3"; }
adj k10_old 0.5
adj k10_new 0.5
adj k10_el  -0.8
adj k10_c1  0.5
adj k10_c2  0.5
for f in k10_old k10_new k10_el k10_c1 k10_c2; do
  printf '%s ' $f
  ffmpeg -hide_banner -i $OUT/$f.mp3 -af ebur128 -f null - 2>&1 | grep 'I:' | tail -1
done
