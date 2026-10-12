#!/bin/bash
# 用 Qwen（现线上引擎）生成与 GS 完全相同的 6 段文案，供同文案对比。
# ⚠️ 文案全部取自 hold 案例（未进训练集），所以对 Qwen 和 GS 都诚实。
set -u
OUT=/mnt/tts/projects/gpt-sovits/out
PY=/mnt/tts/projects/qwen3-tts/venv/bin/python

run(){
  local txt=$1 voice=$2 pre=$3
  echo "=== $voice <- $(basename $txt)"
  $PY /mnt/tts/projects/qwen3-tts/tools/qwen_cmp_gen.py "$txt" "$voice" "$pre" 2>&1 | tail -4
}

# 女声（现线上标杆 = EL_female）
run "$OUT/text_sola_000.txt" EL_female cf000
run "$OUT/text_sola_150.txt" EL_female cf150
run "$OUT/text_sola_300.txt" EL_female cf300

# 男声（现线上标杆 = EL_male）
run "$OUT/text_yohan_000.txt" EL_male cm000
run "$OUT/text_yohan_150.txt" EL_male cm150
run "$OUT/text_yohan_300.txt" EL_male cm300

echo ALLDONE
