#!/bin/bash
# 盲听切片响度对齐：只用线性 volume，不用 loudnorm（不改动态）
# EL 标杆 -24.2，我们 -25.5，差 1.3dB 会让员工把"响"误当成"像真人"
set -e
OUT=/mnt/tts/data/tone-ab
cd /tmp
adj() { ffmpeg -y -v error -i "$OUT/$1.mp3" -af "volume=$2dB" /tmp/a_$1.mp3; \
        echo 'a19910119' | sudo -S cp /tmp/a_$1.mp3 "$OUT/$1.mp3" >/dev/null 2>&1 || \
        sudo cp /tmp/a_$1.mp3 "$OUT/$1.mp3"; }

adj k9_old 0.5
adj k9_new 0.5
adj k9_el  -0.8
adj k9_c1  0.5
adj k9_c2  0.5

for f in k9_old k9_new k9_el k9_c1 k9_c2; do
  printf '%s ' $f
  ffmpeg -hide_banner -i $OUT/$f.mp3 -af ebur128 -f null - 2>&1 | grep 'I:' | tail -1
done
