#!/bin/bash
# d 轮反馈部署 + 音色修复上线（需要 root，用 sudo bash 执行）
#
# 做两件事：
#   1. 部署新题库（热更新，feedback 页每次请求都读 items.json，无需重启）
#      · 备份旧 items.json
#      · 拷入 6 个片段 + 新 items.json
#   2. 重启主服务，让 pipeline.py 的两处修复生效：
#      · CHUNK_MAX_SECONDS 250 → 600（一次成型，音色 0.844 → 0.95 的关键）
#      · cps 钳位 6.0~10.0（防 yohan_male.cps_lock=4.01 那种脏数据把稿子切成 5 段）
#
# ⚠️ 执行前请确认没有员工正在生成任务（当前活跃任务数 0，安全）。
# 回滚：
#   cp /mnt/tts/data/tone-ab/items.json.bak-1013 /mnt/tts/data/tone-ab/items.json
#   把 /mnt/tts/projects/qwen3-tts/app/pipeline.py 里 CHUNK_MAX_SECONDS 默认值改回 250 并重启
set -e

TS=$(date +%m%d)
TONE=/mnt/tts/data/tone-ab
SRC=/tmp/fbclipd

echo "== 1. 备份旧题库 =="
ls -la "$TONE/items.json"
cp -a "$TONE/items.json" "$TONE/items.json.bak-$TS"
echo "   -> $TONE/items.json.bak-$TS"

echo
echo "== 2. 部署新片段 + 新题库 =="
ls -1 "$SRC"/out/fd_*.mp3
cp "$SRC"/out/fd_*.mp3 "$TONE"/
cp "$SRC"/items_20261013a.json "$TONE"/items.json
chmod 644 "$TONE"/fd_*.mp3 "$TONE"/items.json
echo "   -> 片段 $(ls -1 "$TONE"/fd_*.mp3 | wc -l) 个"

echo
echo "== 3. 重启主服务（加载分块修复 + cps 钳位）=="
systemctl restart qwen-tts
sleep 8
systemctl is-active qwen-tts

echo
echo "== 4. 冒烟：服务健康 + 题库可读 =="
curl -s --max-time 10 http://127.0.0.1:8000/health | head -c 400
echo
curl -s --max-time 10 http://127.0.0.1:8000/api/fb/items \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('题目数:', len(d.get('items',[]))); [print('  ', i['id'], '|', [(o['key'],o['file']) for o in i['options']]) for i in d.get('items',[])]"

echo
echo "完成。"
