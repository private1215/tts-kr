#!/usr/bin/env bash
# 🔴 切换反馈台出题目录（只需 sudo 跑一次，之后不再需要）
#
# 为什么需要这一步（运维死循环）：
#   · /mnt/tts/data/tone-ab 属主是 root，出题脚本以 zheng 跑会 Permission denied
#   · 改 systemd 的 TTS_FB_DIR 要 sudo，而 sudo 密码已按安全要求脱敏
#   · 重启 qwen-tts 也要 sudo
#   → 结果「想更新题库」被卡死在权限上。
#
# 解法（已在 server_new.py 里实现，验证过语法与目录选择逻辑）：
#   服务启动时按 items.json 的 mtime 排序候选目录，选最新的生效。
#   出题脚本写进 /mnt/tts/fb（zheng 可写）即自动生效，
#   **之后更新题库不再需要 sudo，也不需要重启**。
#
# 用法：
#   ssh yuyin 'echo "密码" | sudo -S -p "" bash /mnt/tts/projects/qwen3-tts/deploy/apply_fbdir.sh'
set -euo pipefail
APP=/mnt/tts/projects/qwen3-tts/app

echo "1/3 检查员工投票历史（切换后必须仍在）"
BEFORE=$(wc -l < /mnt/tts/data/tone-ab/votes.jsonl)
AFTER=$(wc -l < /mnt/tts/fb/votes.jsonl)
echo "    旧目录 ${BEFORE} 行 / 新目录 ${AFTER} 行"
if [ "$BEFORE" != "$AFTER" ]; then
  echo "    🔴 投票历史不一致，**中止**（不要切换）"
  exit 1
fi
echo "    ✅ 一致（${BEFORE} 条）"

echo "2/3 换上新代码"
cp "${APP}/server_new.py" "${APP}/server.py"
python3 -c "import ast;ast.parse(open('${APP}/server.py',encoding='utf-8').read())"
echo "    ✅ server.py 已更新且语法通过"

echo "3/3 重启 qwen-tts（当前无员工任务，可安全重启）"
systemctl restart qwen-tts
sleep 6

echo
echo "=== 验证：打这个端点确认生效目录 ==="
curl -s http://127.0.0.1:8000/api/fb/where || {
  echo "（服务可能还在启动，稍等几秒再打）"
  echo "  curl -s http://127.0.0.1:8000/api/fb/where"
}
echo
echo "完成。此后改题库直接跑 deploy/mk_o22.py 即可，不需要 sudo。"