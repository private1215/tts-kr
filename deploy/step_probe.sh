#!/bin/bash
# 分步定位：变调三步(asetrate -> aresample -> atempo)里哪一步把 60-150Hz 削没了
# 基线 nof0 60-150=0.0322；完整三步后 0.0065
set -e
cd /mnt/tts/diag/x8197
SR=16000
R=1.2130
NEW=$(python3 -c "print(int(round($SR*$R)))")
SRC=m_nof0.wav   # 未变调、已清洗+EQ，作为干净起点

ffmpeg -y -v error -i $SRC -t 60 -af "asetrate=$NEW"                 s1.wav
ffmpeg -y -v error -i $SRC -t 60 -af "asetrate=$NEW,aresample=$SR"   s2.wav
ffmpeg -y -v error -i $SRC -t 60 -af "atempo=$(python3 -c "print(1/$R)")" s3.wav
ffmpeg -y -v error -i $SRC -t 60 -af "asetrate=$NEW,aresample=$SR,atempo=$(python3 -c "print(1/$R)")" s4.wav
ffmpeg -y -v error -i $SRC -t 60 -af "atempo=$(python3 -c "print(1/$R)"),asetrate=$NEW,aresample=$SR" s5.wav
ls -la s1.wav s2.wav s3.wav s4.wav s5.wav
