#!/bin/bash
# 对照音频时长核对：语速(sps)必须由「时长」佐证，否则可能是静音污染
GS=/mnt/tts/projects/gpt-sovits/out
for f in "$GS"/ref_sola_00b6a45a.wav "$GS"/gs_sola_00b6a45a_ft_e40.wav \
         "$GS"/gs_sola_00b6a45a_zeroshot.wav "$GS"/ref_yohan_1a6c20a6.wav \
         "$GS"/gs_yohan_1a6c20a6_ft_e40.wav "$GS"/gs_yohan_1a6c20a6_zeroshot.wav \
         /mnt/tts/output/gs-cmp-EL_female/gs-cmp-EL_female-1.mp3 \
         /mnt/tts/output/gs-cmp-SOLA_female/gs-cmp-SOLA_female-1.mp3 \
         /mnt/tts/output/gs-cmp-m-EL_male/gs-cmp-m-EL_male-1.mp3 \
         /mnt/tts/output/gs-cmp-m-yohan_male/gs-cmp-m-yohan_male-1.mp3; do
  d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f" 2>/dev/null)
  printf "%-58s %s\n" "$(basename "$f")" "${d:-缺失}"
done
