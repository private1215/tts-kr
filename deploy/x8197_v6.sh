#!/bin/bash
# 产出第十四轮盲听用的男声候选（x8197，与 EL 同文案，可直接对标杆）
#   base_p : 变调（F0 对齐 EL 197.7）后**裸片**，供 EQ 网格搜索
#   base_n : 不变调裸片，供 EQ 网格搜索（对照组：检验变调到底是不是净收益）
# 两者除 --no-pitch 外链路完全一致，保证 A/B 是单变量。
set -e
cd /mnt/tts/diag/x8197
PY=/mnt/tts/projects/qwen3-tts/venv/bin/python
FC=/mnt/tts/projects/qwen3-tts/app/fix_chain.py
CH=$(wc -m < text.txt)
echo "字数: $CH"

$PY $FC vox_male_raw.wav base_p.wav --gender m --target-cps 8.37 --chars $CH --eq anull
$PY $FC vox_male_raw.wav base_n.wav --gender m --target-cps 8.37 --chars $CH --eq anull --no-pitch
ls -la base_p.wav base_n.wav
