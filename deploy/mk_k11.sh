#!/bin/bash
# 第十三轮切片：含「开头 8 秒」专项
set -e
cd /mnt/tts/diag
OUT=/mnt/tts/data/tone-ab
S=40; T=35; H=8

cut() { # cut <in> <name> <start> <dur>
  ffmpeg -y -v error -ss $3 -t $4 -i "$1" -ar 44100 -b:a 128k /tmp/$2.mp3
  echo "${TTS_SUDO_PW:?需要设置 TTS_SUDO_PW}" | sudo -S -p "" cp /tmp/$2.mp3 "$OUT/$2.mp3" >/dev/null 2>&1 || \
    sudo cp /tmp/$2.mp3 "$OUT/$2.mp3"
}

cut x8197/m_v4.wav k11_old $S $T
cut x8197/m_v5.wav k11_new $S $T
cut /mnt/tts/data/cases/8197d9ac8a35/merged.wav k11_el $S $T
cut x8197/m_v5.wav k11_c1 $S $T
cut x8197/m_v5.wav k11_c2 $S $T
cut x8197/m_v5.wav k11_hnew $S $H
cut /mnt/tts/data/cases/8197d9ac8a35/merged.wav k11_hel $S $H

echo "=== LUFS ==="
for f in k11_old k11_new k11_el k11_c1 k11_c2 k11_hnew k11_hel; do
  printf '%-10s ' $f
  ffmpeg -hide_banner -i $OUT/$f.mp3 -af ebur128 -f null - 2>&1 | grep 'I:' | tail -1
done
