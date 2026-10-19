#!/bin/bash
# 手工强制重训（2026-10-17 白天，用户确认服务器空闲、测试阶段无正式使用）
#
# 为什么要手工跑而不是等夜训：
#   gs_auto_train.py --all --dry-run 报「SOLA / yohan 无变化，跳过」，
#   但实测 dataset_SOLA(10-16 20:37) / dataset_yohan(10-16 09:05) 都**比
#   现有权重新**（SOLA e40=07:56、yohan e40=09:52）——「没变化」判据漏判了。
#   结果：第48轮加的 epoch 自适应（yohan 6160/119 → e52）**一直没有真正生效**。
#
# 目标：
#   1. yohan 组用 e52（6188 步，原 e40 只有 4840 步）→ 验证男声 click 是否下降
#   2. SOLA 组用 180 分钟预算的 174.3 分钟训练集（原 160 分钟会丢 ea3154b3）
#
# 顺序：prep（重建 train.list）→ preprocess（特征提取）→ train（s2 微调）
# 每组 40~60 分钟，后台跑，用 setsid 防 ssh 会话结束被杀。
cd /mnt/tts/projects/qwen3-tts || exit 1
LOG=/mnt/tts/projects/gpt-sovits/manual_train_20261017.log
{
  echo "===== $(date '+%F %T') 手工强制重训启动 ====="
  echo "-- 前置：确认无其它训练在跑 --"
  pgrep -af "s2_train.py|gs_run_group.sh" || echo "  无"

  for SPK in yohan SOLA; do
    echo "===== $(date '+%F %T') [$SPK] prep ====="
    venv/bin/python -u tools/gs_prep_group.py --spk "$SPK" \
        --max-minutes 180 --exclude-cer 0.35 2>&1 | tail -12
    echo "===== $(date '+%F %T') [$SPK] preprocess ====="
    bash tools/gs_run_group.sh "$SPK" preprocess 2>&1 | tail -25
    echo "===== $(date '+%F %T') [$SPK] train ====="
    bash tools/gs_run_group.sh "$SPK" train 2>&1 | tail -40
    echo "===== $(date '+%F %T') [$SPK] 完成 rc=$? ====="
  done

  echo "===== $(date '+%F %T') 让 GS 服务卸载旧权重 ====="
  curl -s -X POST http://127.0.0.1:8200/unload; echo
  echo "===== $(date '+%F %T') 全部结束 ====="
} >> "$LOG" 2>&1
