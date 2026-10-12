#!/bin/bash
# 核实：GS「微调版」与「零样本版」是不是同一个文件。
# 为什么必须查：两者文件大小/时长完全相同（女 33.58s / 男 59.80s），
# 若实为同一份输出，则「微调 vs 零样本」的对照结论全部作废。
GS=/mnt/tts/projects/gpt-sovits/out
for f in "$GS"/gs_sola_00b6a45a_ft_e40.wav "$GS"/gs_sola_00b6a45a_zeroshot.wav \
         "$GS"/gs_yohan_1a6c20a6_ft_e40.wav "$GS"/gs_yohan_1a6c20a6_zeroshot.wav; do
  printf "%-40s %s\n" "$(basename "$f")" "$(md5sum "$f" | cut -d' ' -f1)"
done
