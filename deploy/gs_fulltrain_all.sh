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

# 特征是否已就绪：2-name2text-0.txt 条数对得上 + 三个特征目录都非空。
# 为什么要这个判断：preprocess 跑一次要 4 分钟，重跑纯浪费；而且 exp 目录
# 一旦被移走，已提取的 HuBERT/bert 特征全部作废。
feat_ready(){  # $1=spk
  local spk=$1 exp="$GS/exp/${spk}_v2pro"
  local n_list n_txt
  n_list=$(wc -l < "$GS/dataset_${spk}/train.list" 2>/dev/null || echo 0)
  [ "$n_list" = "0" ] && return 1
  n_txt=$(wc -l < "$exp/2-name2text-0.txt" 2>/dev/null || echo 0)
  [ "$n_txt" != "$n_list" ] && return 1
  [ -n "$(ls "$exp/4-cnhubert" 2>/dev/null | head -1)" ] || return 1
  [ -n "$(ls "$exp/5-wav32k" 2>/dev/null | head -1)" ] || return 1
  [ -n "$(ls "$exp/7-sv_cn" 2>/dev/null | head -1)" ] || return 1
  return 0
}

train_group(){  # $1=spk
  local spk=$1
  local exp="$GS/exp/${spk}_v2pro"
  local n_list
  n_list=$(wc -l < "$GS/dataset_${spk}/train.list")

  if feat_ready "$spk"; then
    say "✅ ${spk} 特征已就绪（2-name2text-0.txt=${n_list} 条），跳过 preprocess"
  else
    # 🔴🔴 必须清掉含旧 checkpoint 的 exp 目录，否则训练会被旧 checkpoint 接管
    #     （2026-10-12 踩过）：s2_train.py 发现 logs_s2_*/D_*.pth 会打印
    #     "start training from epoch 41" 然后**立刻结束**（配置 epochs=40），
    #     一条新权重都不产出，而总控还认为"训练完成"——完全静默的失败。
    #     dataloader 也会退回首轮的条数（实测 1034 → 420）。
    #     旧权重实际在 exp/<spk>_v2pro/logs_s2_v2ProPlus/{D,G}_*.pth。
    if [ -d "$exp" ]; then
      local bak="$GS/exp/_old_${spk}_v2pro_$(date +%m%d_%H%M%S)"
      mv "$exp" "$bak"
      say "⚠️ 移走旧 exp 目录 -> $bak（防止从旧 epoch 续跑 / 特征不完整重提）"
    fi
    say "########## ${spk} preprocess ##########"
    if ! bash "$PROJ/tools/gs_run_group.sh" "$spk" preprocess >>"$LOG" 2>&1; then
      say "❌ ${spk} preprocess 失败"; return 1
    fi
    # 特征产物自检。🔴 2026-10-12 教训：分片文件名是 **2-name2text-0.txt**
    #    （i_part=0 的后缀），不是 2-name2text.txt——后者是 train 步才把
    #    2-name2text-*.txt 合并出来的。写错文件名会让 wc 恒为 0，把好的
    #    特征误判成"读旧数据"，白白跳过训练（yohan 就这样被跳过一次）。
    local n_txt
    n_txt=$(wc -l < "$exp/2-name2text-0.txt" 2>/dev/null || echo 0)
    say "特征自检：train.list=${n_list}  2-name2text-0.txt=${n_txt}"
    if [ "$n_txt" != "$n_list" ]; then
      say "❌ 特征条数与数据集不符，放弃训练"; return 1
    fi
  fi

  say "########## ${spk} train ##########"
  if ! bash "$PROJ/tools/gs_run_group.sh" "$spk" train >>"$LOG" 2>&1; then
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

# 🔴 2026-10-12：失败时**绝不写 ALL_DONE**。验证链（verify_chain.sh）靠
#    grep 日志里的 ALL_DONE 触发，旧版无论成败都写，导致训练全挂时验证链
#    照样拿旧权重跑一遍"验证通过"——比不验证更坏。
FAIL=0
if wait_list yohan 5400; then
  train_group yohan || { say "⚠️ yohan 失败，继续下一个"; FAIL=1; }
else
  FAIL=1
fi
if wait_list SOLA 10800; then
  train_group SOLA || { say "⚠️ SOLA 失败"; FAIL=1; }
else
  FAIL=1
fi

say "===== 全部结束，权重目录 ====="
ls -la "$GS/repo/SoVITS_weights_v2ProPlus/" | tee -a "$LOG"
if [ "$FAIL" = "1" ]; then
  say "TRAIN_FAILED（有组失败，不写 ALL_DONE，验证链不会触发）"
  exit 1
fi
say "ALL_DONE"
