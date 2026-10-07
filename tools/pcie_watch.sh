#!/bin/bash
# 3090 PCIe / 供电带载监控：每 10 秒采一次，直到被杀掉。
#
# 看什么
#   - aer_dev_correctable / fatal / nonfatal：累计错误计数（sysfs，重启清零）
#   - nvidia-smi 的 PCIe Generation / Link Width / Replays Since Reset
#   - 功耗 / 温度 / throttle reasons：供电与散热是否撑不住
#   - dmesg 新出现的 AER / Xid
#
# 用法： bash tools/pcie_watch.sh [秒] [输出文件]
#   bash tools/pcie_watch.sh 2100 /mnt/tts/diag/watch_$(date +%m%d_%H%M).log
DEV=0000:04:00.0
RP=0000:00:03.0
SEC=${1:-2100}
OUT=${2:-/mnt/tts/diag/watch.log}
mkdir -p "$(dirname "$OUT")"

echo "watch start $(date '+%F %T') dev=$DEV rp=$RP" | tee -a "$OUT"
echo "t,power_w,temp_c,util_pct,mem_mib,gen,cur_gen,width,cur_width,replays,aer_corr,aer_fatal,aer_nonfatal,throttle" >> "$OUT"

END=$(( $(date +%s) + SEC ))
LAST_DMESG=0
while [ "$(date +%s)" -lt "$END" ]; do
  Q=$(nvidia-smi -i 1 --query-gpu=power.draw,temperature.gpu,utilization.gpu,memory.used,clocks_throttle_reasons.active --format=csv,noheader,nounits 2>/dev/null)
  P=$(cat /sys/bus/pci/devices/$DEV/current_link_speed 2>/dev/null | tr -d ' GT/s')
  W=$(cat /sys/bus/pci/devices/$DEV/current_link_width 2>/dev/null)
  AC=$(cat /sys/bus/pci/devices/$DEV/aer_dev_correctable 2>/dev/null | awk '{s+=$2} END{print s+0}')
  AF=$(cat /sys/bus/pci/devices/$DEV/aer_dev_fatal 2>/dev/null | awk '{s+=$2} END{print s+0}')
  AN=$(cat /sys/bus/pci/devices/$DEV/aer_dev_nonfatal 2>/dev/null | awk '{s+=$2} END{print s+0}')
  # 根端口侧同理
  AC2=$(cat /sys/bus/pci/devices/$RP/aer_rootport_total_err_cor 2>/dev/null | awk '{s+=$2} END{print s+0}')
  RP_INFO=$(nvidia-smi -i 1 --query-gpu=pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  REPLAY=$(nvidia-smi -q -i 1 2>/dev/null | grep -i "Replays Since Reset" | head -1 | awk -F': ' '{print $2}' | tr -d ' ')
  echo "$(date '+%H:%M:%S'),$Q,${RP_INFO},${AC:-0},${AF:-0},${AN:-0},rp_cor=${AC2:-0}" >> "$OUT"
  # dmesg 增量（只看新行）
  NEW=$(dmesg 2>/dev/null | grep -icE "aer|xid|fallen off" || true)
  if [ "$NEW" != "$LAST_DMESG" ]; then
    dmesg 2>/dev/null | grep -iE "aer|xid|fallen off" | tail -5 >> "$OUT"
    LAST_DMESG=$NEW
  fi
  sleep 10
done
echo "watch end $(date '+%F %T')" >> "$OUT"
