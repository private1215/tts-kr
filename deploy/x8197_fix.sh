#!/bin/bash
# 跨文案音高验证（8197 有同文本 EL，走最准的 --el 逐窗对齐）
set -e
PY=/mnt/tts/projects/omnivoice/venv/bin/python
cd /mnt/tts/diag

# 🔴 不用 --el 逐窗对齐：那个模式对的是「音节核/秒」，跨引擎不抵消量纲差。
#    实测 x8197 走 --el 后是 8.94 字/秒，而 EL 同篇是 8.37 —— 快了 6.8%。
#    一律用 --target-cps（字/秒），这个才是跨文案/跨引擎可比。
echo "== 对照组：不做 F0 对齐 =="
$PY fix_chain.py x8197/vox_male_raw.wav x8197/m_nof0.wav \
  --gender m --target-cps 8.37 --chars 4489 --no-pitch

echo "== 实验组：自动 F0 对齐到 197.7 =="
$PY fix_chain.py x8197/vox_male_raw.wav x8197/m_f0.wav \
  --gender m --target-cps 8.37 --chars 4489

echo "== cf9（员工那条真实文案，自动 F0 对齐）=="
$PY fix_chain.py cf9/vox_male_raw.wav cf9/m_f0.wav \
  --gender m --target-cps 8.37 --chars 3919
echo ALLDONE
