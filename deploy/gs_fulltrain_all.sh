#!/usr/bin/env bash
# 全量微调总控：等数据准备 → preprocess → train，两组串行（都抢 3090）。
#
# 为什么串行：3090 上 Cosy 常驻 ~2.2G，训练峰值 ~10G。两个训练并行必 OOM。
# 为什么 yohan 先：它的数据量小（100 分钟 vs 160），先出结果先验证。
#
# 用法（后台跑，日志自动落盘）：
#   setsid nohup bash deploy/gs_fulltrain_all.sh > /tmp/gs_fulltrain_all.log 2>&1 &
set -u

PROJ=/mnt/tts/projects/qwen3-tts
GS=/mnt/tts/projects/gpt-sovits
VENV=$GS/venv/bin/python
LOG=/tmp/gs_fulltrain_all.log

say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

wait_list(){  # $1=spk  $2=超时秒
  local spk=$1 t0=$(date +%s)
  while [ ! -f "$GS/dataset_${spk}/train.list" ]; do
    sleep 30
    if [ $(( $(date +%s) - t0 )) -ge "$2" ]; then
      say "❌ 等 ${spk} 数据超时（$2s），放弃"; return 1
    fi
  done
  say "✅ ${spk} 数据就绪：$(wc -l < "$GS/dataset_${spk}/train.list") 条"
  return 0
}

train_group(){  # $1=spk
  local spk=$1
  say "########## ${spk} preprocess ##########"
  if ! bash "$PROJ/deploy/gs_run_group.sh" "$spk" preprocess >>"$LOG" 2>&1; then
    say "❌ ${spk} preprocess 失败"; return 1
  fi
  say "########## ${spk} train ##########"
  if ! bash "$PROJ/deploy/gs_run_group.sh" "$spk" train >>"$LOG" 2>&1; then
    say "❌ ${spk} train 失败"; return 1
  fi
  # 🔴 必须让 GS 服务丢掉内存里的旧权重，否则新权重文件躺那儿不生效（静默失败）
  curl -s -m 20 -X POST http://127.0.0.1:8200/unload >/dev/null \
    && say "✅ GS 服务已 unload，下次请求加载新权重" \
    || say "⚠️ GS unload 失败（不影响产物，手动重启即可）"
  say "✅ ${spk} 训练完成"
  return 0
}

say "===== 全量微调总控启动 ====="
say "等待 yohan / SOLA 数据准备完成（gs_prep_group.py 正在跑）"

if wait_list yohan 5400; then
  train_group yohan || say "⚠️ yohan 失败，继续下一个"
fi
if wait_list SOLA 10800; then
  train_group SOLA || say "⚠️ SOLA 失败"
fi

say "===== 全部结束，权重目录 ====="
ls -la "$GS/repo/SoVITS_weights_v2ProPlus/" | tee -a "$LOG"
say "ALL_DONE"
