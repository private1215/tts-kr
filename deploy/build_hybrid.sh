#!/bin/sh
# 混合变调链：asetrate 只搬一点点（管频谱/共振峰），音高交给分窗 PSOLA。
#
# 为什么（2026-10-14）：
#   现役 asetrate ×1.21 把**整个频谱**搬高 21%，直接造成
#     b2000_4000 +12.5sigma / b4000_8000 +18.8sigma（高频严重过量）
#   之后又必须用 EQ 猛切高频，而猛切高频把 flat 打到 -5.2sigma（发假）。
#   另一头，PSOLA 完全不搬频谱 -> 高频反而 -6.31sigma（不够）。
#   EL 在两者之间：按 z 线性插值，搬 4~5% 时 b4000_8000 约 -1.5sigma，
#   同时 F2 从裸片 1495 升到 ~1560（EL 1556）—— **一个参数同时修两处**。
#
# 步骤（时长全程不变）：
#   1) atempo 把裸片拉到 EL 语速（469.76 -> 534.63s）
#   2) asetrate R + aresample + atempo 1/R  —— 只搬频谱与基频，不改时长
#   3) PSOLA 分窗把基频拉回 EL 的 197.7Hz —— 不动共振峰、不改时长
set -e
D=/mnt/tts/diag/x8197
PY=/mnt/tts/projects/qwen3-tts/venv/bin/python
R=${1:-1.045}

cd "$D"
echo "R=$R"
ffmpeg -y -v error -i vox_male_raw.wav \
  -af "atempo=0.87835" -ar 24000 -ac 1 _hb_rate.wav
ffmpeg -y -v error -i _hb_rate.wav \
  -af "asetrate=24000*$R,aresample=24000,atempo=1/$R" \
  -ar 24000 -ac 1 _hb_spec.wav
$PY psola_shift.py _hb_spec.wav m_hb_$R.wav --target-f0 197.7 --windowed
ffprobe -v error -show_entries format=duration \
  -of default=nw=1:nk=1 m_hb_$R.wav
