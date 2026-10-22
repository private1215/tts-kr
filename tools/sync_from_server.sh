#!/usr/bin/env bash
# sync_from_server.sh —— 把服务端代码同步回本地仓库。
#
# 🔴 为什么需要它
#   服务端 /mnt/tts/projects/qwen3-tts **不是 git 仓库**，是真相源。
#   在服务端改完不同步 = 改动永久丢失（2026-10-22 曾一次发现 25 个文件不一致，
#   且 omnivoice/ 整个项目从未入库）。
#
# 🔴 用法（收工前必跑）
#   cd <本地仓库根>
#   bash tools/sync_from_server.sh            # 同步 + 报告
#   git add -A && git commit -m "..." && git push
#
# 🔴 只同步代码，不整目录 rsync（服务端 app/ 是真相，按文件 scp）。
#   systemd drop-in（/etc/systemd/system/*.service.d/）要 sudo 才能读，手动归档：
#     ssh yuyin "cat /etc/systemd/system/omni-tts.service.d/20-steps48.conf" \
#       > deploy/systemd/20-steps48.conf
set -eo pipefail

SERVER="${SERVER:-yuyin}"
REMOTE="${REMOTE:-/mnt/tts/projects/qwen3-tts}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

echo "==> 服务端 $SERVER:$REMOTE  本地 $HERE"

# 1) 取服务端清单（含 md5），另加 omnivoice（不在 qwen3-tts 仓库里）
#    每行三列：md5 <本地相对路径> <服务端绝对路径>
LIST="$(ssh "$SERVER" "cd '$REMOTE' && md5sum app/*.py tools/*.py 2>/dev/null" \
        | sed "s|^\([0-9a-f]*\)  *\(.*\)|\1  \2  $REMOTE/\2|")"
OMNI="$(ssh "$SERVER" "cd /mnt/tts/projects/omnivoice && md5sum omni_server.py 2>/dev/null" \
        | sed "s|^\([0-9a-f]*\)  *\(.*\)|\1  omnivoice/\2  /mnt/tts/projects/omnivoice/\2|" || true)"
if [ -n "$OMNI" ]; then
    LIST="$LIST
$OMNI"
fi

TMP="$(mktemp)"
echo "$LIST" | while read -r h p rp; do
    [ -z "${p:-}" ] && continue
    if [ -f "$p" ]; then
        l="$(md5 -q "$p" 2>/dev/null || md5sum "$p" | awk '{print $1}')"
    else
        l="MISSING"
    fi
    if [ "$l" = "$h" ]; then
        continue
    fi
    mkdir -p "$(dirname "$p")"
    scp -q "$SERVER:$rp" "$p"
    if [ "$l" = "MISSING" ]; then
        echo "新增 $p"
    else
        echo "更新 $p"
    fi
done | tee "$TMP"

n_sync="$(grep -c '^更新 ' "$TMP" || true)"
n_new="$(grep -c '^新增 ' "$TMP" || true)"
n_total="$(echo "$LIST" | grep -c '[^[:space:]]' || true)"
rm -f "$TMP"

echo "==> 共比对 $n_total 个文件：更新 ${n_sync:-0}、新增 ${n_new:-0}、一致 $(( ${n_total:-0} - ${n_sync:-0} - ${n_new:-0} ))"

if [ "${n_sync:-0}" -gt 0 ] || [ "${n_new:-0}" -gt 0 ]; then
    echo "==> 有差异，记得提交："
    echo "    git add -A && git commit -m '同步服务端代码' && git push"
else
    echo "==> 已是最新，无需同步"
fi
