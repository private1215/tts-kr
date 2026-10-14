#!/bin/bash
# 二分定位：这段「碎」是引擎(raw)自带的，还是修复链弄出来的？
# 成品 m_v4 534.7s 对应 raw 469.76s，时间轴按 0.8785 折算：40~48s -> 35.1~42.2s
set -e
cd /mnt/tts/diag
mkdir -p headseg
ffmpeg -y -v error -ss 35.1 -t 7.1 -i x8197/vox_male_raw.wav headseg/raw_h.wav
# 对照组（未变调、同样清洗）
ffmpeg -y -v error -ss 40 -t 8 -i x8197/m_nof0.wav headseg/nof0_h.wav
# 跳过清洗的版本：单独跑一次 --no-clean 太长，先看 no-clean 的旧产物
ls -la headseg/
