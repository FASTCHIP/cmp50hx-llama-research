#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/home/fastchip/cmp50hx-llama-research
ROLLBACK_UNIT="$ROOT/rollback/llama-qwen.service.initial"
LOG="$ROOT/logs/campaign.log"
LABEL=${1:?label}; shift
[[ ${1:-} == -- ]] && shift
exec 9>>"$ROOT/logs/trace-${LABEL}.log"; BASH_XTRACEFD=9; set -x
msk(){ TZ=Europe/Moscow date --iso-8601=seconds; }
log(){ printf '%s %s\n' "$(msk)" "$*" >>"$LOG"; }
restore(){
  rc=$?
  trap - EXIT INT TERM
  log "variant=$LABEL restore_begin rc=$rc"
  sudo -n systemctl stop llama-research.service 2>/dev/null || true
  sudo -n cp "$ROLLBACK_UNIT" /etc/systemd/system/llama-qwen.service
  sudo -n systemctl daemon-reload
  sudo -n systemctl restart llama-qwen.service
  "$ROOT/harness/wait_health.sh" http://127.0.0.1:8081/health 480 llama-qwen.service
  # Research owns GPU0-2; the separate GPU3 Bonsai service stays stopped during campaign.
  n=$(pgrep -x llama-server | wc -l)
  [[ "$n" -eq 1 ]] || { log "restore_FAIL llama_process_count=$n"; return 1; }
  for b in 01:00.0 02:00.0 04:00.0; do
    [[ $(cat /sys/bus/pci/devices/0000:$b/current_link_speed) == '5.0 GT/s PCIe' ]]
    [[ $(cat /sys/bus/pci/devices/0000:$b/current_link_width) == 8 ]]
  done
  log "variant=$LABEL restore_PASS health=200 llama_process_count=1"
  return "$rc"
}
trap restore EXIT INT TERM

log "variant=$LABEL preguard_begin"
free_k=$(df -Pk / | awk 'NR==2{print $4}')
(( free_k > 15*1024*1024 )) || { log "hard_stop root_free_k=$free_k"; exit 90; }
[[ $(nvidia-smi -L | grep -c 'CMP 50HX') -eq 4 ]] || { log 'hard_stop gpu_count'; exit 91; }
for b in 01:00.0 02:00.0 04:00.0; do
  [[ $(cat /sys/bus/pci/devices/0000:$b/current_link_speed) == '5.0 GT/s PCIe' ]]
  [[ $(cat /sys/bus/pci/devices/0000:$b/current_link_width) == 8 ]]
done
errs=$(sudo -n journalctl -k -b --no-pager | grep -cE 'NVRM.*[Xx]id [0-9]+|NV_ERR_RESET_REQUIRED|AER:.*(Uncorrected|Corrected error)|RmInitAdapter|Booter failed' || true)
[[ "$errs" -eq 0 ]] || { log "hard_stop kernel_errors=$errs"; exit 92; }
sudo -n systemctl stop bonsai-2.service 2>/dev/null || true
sudo -n systemctl stop llama-qwen.service
until ! ss -tln | grep -q ':8081 '; do sleep 1; done
sudo -n systemctl reset-failed llama-research.service 2>/dev/null || true
log "variant=$LABEL launch cmd=$*"
sudo -n systemd-run --unit=llama-research.service --service-type=simple --uid=fastchip --gid=fastchip \
  --property=Restart=no --property=TimeoutStopSec=20 --property=KillMode=mixed \
  --setenv=CUDA_VISIBLE_DEVICES=0,1,2 --setenv=GGML_CUDA_GRAPH_OPT=1 \
  --setenv=NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0} --setenv=NCCL_P2P_LEVEL=PHB \
  --setenv=LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/home/fastchip/llama.cpp-vbbb-b7d4d85-sm75/bin:/home/fastchip/llama.cpp-vbbb-b7d4d85-sm75/lib:/usr/local/cuda-12.8/lib64} \
  "$@"
"$ROOT/harness/wait_health.sh" http://127.0.0.1:8081/health "${HEALTH_TIMEOUT:-600}" llama-research.service
[[ $(pgrep -x llama-server | wc -l) -eq 1 ]]
log "variant=$LABEL health_PASS"
if [[ -n ${BENCH_CMD:-} ]]; then bash -lc "$BENCH_CMD"; fi
log "variant=$LABEL bench_PASS"
