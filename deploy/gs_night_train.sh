#!/bin/bash
# 夜间全量微调（避开员工白天使用）
#
# 为什么放在晚上：训练要抢 3090，而员工白天生成也走 3090（CosyVoice3 常驻）。
# 2026-10-13 实测：数据准备（VAD 切片）跑着的时候，线上生成任务从 8 分钟拖到
# 33 分钟 —— 白天跑训练会让员工明显变慢，所以只能排到下班后。
#
# 为什么脚本要进仓库而不是 /tmp：服务器 /tmp 会被清，历史上 fidelity3.py /
# eval5r20.py 就这么丢过一次，r20 基线一度无法复现。
#
# 用法（已在服务器上这样装）：
#   (crontab -l | grep -v gs_night; \
#    echo "0 19 * * * /bin/bash <repo>/deploy/gs_night_train.sh") | crontab -
cd /mnt/tts/projects/qwen3-tts || exit 1
LOG=/mnt/tts/projects/gpt-sovits/auto_train.log
{
  echo "===== $(date '+%F %T') 夜间全量微调启动 ====="
  venv/bin/python -u tools/gs_prep_group.py --spk SOLA \
      --max-minutes 160 --exclude-cer 0.35 --hold 00b6a45ad6aa 2>&1 | tail -20
  echo "---- SOLA prep rc=$? $(date '+%T') ----"
  venv/bin/python -u tools/gs_auto_train.py --all 2>&1 | tail -60
  echo "===== $(date '+%F %T') 结束 rc=$? ====="
} >> "$LOG" 2>&1
