#!/usr/bin/env bash
# 训练完成后的验证链：等 ALL_DONE → 三引擎走线上链路重出长文案 → 同表评分。
#
# 为什么要单独一条链：训练（3090）和线上生成（3060 + 3090 的 Cosy）会互相拖慢，
# 所以必须等训练彻底结束再跑。这条链只负责"等 + 跑"，失败不重试（重试会叠加显存压力）。
#
# 用法（后台）：
#   setsid nohup bash deploy/verify_chain.sh > /tmp/verify_chain.log 2>&1 &
set -u

PROJ=/mnt/tts/projects/qwen3-tts
VENV=$PROJ/venv/bin/python
LOG=/tmp/verify_chain.log

say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "===== 验证链启动，等待训练总控结束 ====="

# 等 gs_fulltrain_all.log 出现 ALL_DONE（训练链自己写的标记）
t0=$(date +%s)
while ! grep -q ALL_DONE /tmp/gs_fulltrain_all.log 2>/dev/null; do
  sleep 60
  if [ $(( $(date +%s) - t0 )) -ge 14400 ]; then
    say "❌ 等训练超时（4 小时），放弃"; exit 1
  fi
done
say "✅ 训练链已完成"

say "########## 三引擎走线上链路重出长文案 ##########"
if ! $VENV $PROJ/tools/engine_full_cmp.py >>"$LOG" 2>&1; then
  say "❌ 线上链路生成失败"; exit 1
fi

say "########## 生成长文案评分清单 ##########"
$VENV $PROJ/tools/spec_full.py >>"$LOG" 2>&1

say "########## 女声总评 ##########"
$VENV $PROJ/tools/engine_score_all.py \
  --spec $PROJ/tools/score_full_f.json \
  --json-out /mnt/tts/diag/score_full_pipeline_f.json >>"$LOG" 2>&1

say "########## 男声总评 ##########"
$VENV $PROJ/tools/engine_score_all.py \
  --spec $PROJ/tools/score_full_m.json \
  --json-out /mnt/tts/diag/score_full_pipeline_m.json >>"$LOG" 2>&1

say "===== 验证链全部结束 ====="
say "VERIFY_DONE"
