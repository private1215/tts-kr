#!/bin/bash
# 定位「开头别扭」：把 40~48s 这一小段单独剪出来对比 EL
set -e
cd /mnt/tts/diag
mkdir -p headseg
ffmpeg -y -v error -ss 40 -t 8 -i x8197/m_v4.wav headseg/new_h.wav
ffmpeg -y -v error -ss 40 -t 8 -i /mnt/tts/data/cases/8197d9ac8a35/merged.wav headseg/el_h.wav
ffmpeg -y -v error -ss 40 -t 8 -i x8197/m_f0.wav headseg/old_h.wav
ls -la headseg/
