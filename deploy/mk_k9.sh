#!/bin/bash
# 第十一轮（音高专项）盲听切片：取同一时间窗，只差音高
set -e
cd /mnt/tts/diag
OUT=/mnt/tts/data/tone-ab
S=40      # 起点：避开开场呼吸/垫乐
T=35      # 35 秒，员工耐心有限（上一轮只听了 6~7 秒就答题）

cut() { ffmpeg -y -v error -ss $S -t $T -i "$1" -ar 44100 -b:a 128k "$2"; }

cut x8197/m_nof0.wav   $OUT/k9_old.mp3   # 对照组：F0 163.0
cut x8197/m_f0.wav     $OUT/k9_new.mp3   # 实验组：F0 198.3（= EL）
cut /mnt/tts/data/cases/8197d9ac8a35/merged.wav $OUT/k9_el.mp3
# 控制题：同一段放两遍
cut x8197/m_f0.wav     $OUT/k9_c1.mp3
cut x8197/m_f0.wav     $OUT/k9_c2.mp3

for f in k9_old k9_new k9_el k9_c1 k9_c2; do
  printf '%s ' $f
  ffprobe -v error -show_entries format=duration -of csv=p=0 $OUT/$f.mp3
done
echo "=== LUFS（要不要对齐就看这个）==="
for f in k9_old k9_new k9_el; do
  printf '%s ' $f
  ffmpeg -v error -i $OUT/$f.mp3 -af ebur128 -f null - 2>&1 | grep -oE 'I: *-?[0-9.]+ LUFS' | tail -1
done
