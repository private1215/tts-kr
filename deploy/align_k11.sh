#!/bin/bash
# 响度对齐：35s 组拉到 -25.4，8s「开头专项」组拉到 -25.4
set -e
OUT=/mnt/tts/data/tone-ab
adj() { ffmpeg -y -v error -i "$OUT/$1.mp3" -af "volume=$2dB" /tmp/c_$1.mp3; \
        echo "${TTS_SUDO_PW:?需要设置 TTS_SUDO_PW}" | sudo -S -p "" cp /tmp/c_$1.mp3 "$OUT/$1.mp3" >/dev/null 2>&1 || \
        sudo cp /tmp/c_$1.mp3 "$OUT/$1.mp3"; }
adj k11_old 0.2
adj k11_new 0.2
adj k11_el  -1.2
adj k11_c1  0.2
adj k11_c2  0.2
adj k11_hnew 0.7
adj k11_hel -0.4
for f in k11_old k11_new k11_el k11_c1 k11_c2 k11_hnew k11_hel; do
  printf '%-10s ' $f
  ffmpeg -hide_banner -i $OUT/$f.mp3 -af ebur128 -f null - 2>&1 | grep 'I:' | tail -1
done
