#!/usr/bin/env bash
# One llama.cpp variant under the stand's isolation rules.
# Guarantees: snapshots what was running, restores exactly that (unit file included),
# and judges GPU faults against a journal cursor instead of the whole boot history.
set -Eeuo pipefail
ROOT=/home/fastchip/cmp50hx-llama-research
H=$ROOT/harness
STATE_DIR=$ROOT/state
LOG=$ROOT/logs/campaign.log
SERVING_UNIT=llama-qwen.service
SIBLINGS=bonsai-2.service
DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
LABEL=${1:?label}; shift
[[ ${1:-} == -- ]] && shift

mkdir -p "$STATE_DIR"
STATE=$STATE_DIR/variant-$LABEL-initial.json
UNIT_SNAP=$STATE_DIR/llama-qwen.service.$LABEL.snapshot

exec 9>>"$ROOT/logs/trace-${LABEL}.log"; BASH_XTRACEFD=9; set -x
msk(){ TZ=Europe/Moscow date --iso-8601=seconds; }
log(){ printf '%s %s\n' "$(msk)" "$*" >>"$LOG"; }
xid_new(){ python3 "$H/rollback_policy.py" xid-new --state "$STATE" 2>/dev/null || echo unknown; }

restore(){
  local rc=$?
  trap - EXIT INT TERM
  log "variant=$LABEL restore_begin rc=$rc"
  sudo -n systemctl stop llama-research.service 2>/dev/null || true
  if [[ -f $UNIT_SNAP ]]; then
    sudo -n cp "$UNIT_SNAP" "/etc/systemd/system/$SERVING_UNIT"
  else
    log "variant=$LABEL restore_warn unit snapshot missing; using installed unit"
  fi
  sudo -n systemctl daemon-reload
  sudo -n systemctl restart "$SERVING_UNIT"
  "$H/wait_health.sh" http://127.0.0.1:8081/health 480 "$SERVING_UNIT" || log "variant=$LABEL restore_health_FAIL"
  while read -r svc; do
    [[ -z $svc ]] && continue
    sudo -n systemctl start "$svc" || log "variant=$LABEL sibling_start_FAIL svc=$svc"
  done < <(python3 "$H/rollback_policy.py" restore-list --state "$STATE" 2>/dev/null || true)
  sleep 5
  local health count xid
  health=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 http://127.0.0.1:8081/health || echo 000)
  count=$(pgrep -x llama-server | wc -l)
  xid=$(xid_new)
  log "variant=$LABEL restore_done health=$health llama_count=$count new_xid=$xid"
  if [[ $xid =~ ^[0-9]+$ ]] && (( xid > 0 )); then
    log "variant=$LABEL NEW_XID_DETECTED count=$xid"
    python3 "$H/rollback_policy.py" xid-lines --state "$STATE" >>"$ROOT/logs/xid-$LABEL.log" 2>/dev/null || true
  fi
  return $rc
}
trap restore EXIT INT TERM

log "variant=$LABEL preguard_begin devices=$DEVICES"
free_k=$(df -Pk / | awk 'NR==2{print $4}')
(( free_k > 15*1024*1024 )) || { log "hard_stop root_free_k=$free_k"; exit 90; }
[[ $(nvidia-smi -L | grep -c 'CMP 50HX') -eq 4 ]] || { log 'hard_stop gpu_count'; exit 91; }
# snapshot the initial state BEFORE stopping anything: services, journal cursor, exact unit file
python3 "$H/rollback_policy.py" snapshot --services "$SERVING_UNIT,$SIBLINGS" --out "$STATE" >>"$ROOT/logs/snapshot-$LABEL.log" 2>&1 || { log 'hard_stop snapshot_failed'; exit 94; }
if [[ -f /etc/systemd/system/$SERVING_UNIT ]]; then
  sudo -n cp "/etc/systemd/system/$SERVING_UNIT" "$UNIT_SNAP"
  sha256sum "$UNIT_SNAP" | awk '{print "unit_sha16="$1}' | cut -c1-24 >>"$ROOT/logs/snapshot-$LABEL.log"
fi
log "variant=$LABEL snapshot services=$(python3 -c "import json,sys;print(json.load(open('$STATE'))['services'])" 2>/dev/null || echo '?')"
# kernel-level faults that must block the run (Xid history is handled by the cursor, not here)
fatal=$(sudo -n journalctl -k -b --no-pager | grep -cE 'NV_ERR_RESET_REQUIRED|RmInitAdapter|Booter failed|AER:.*Uncorrected' || true)
[[ "$fatal" -eq 0 ]] || { log "hard_stop kernel_fatal=$fatal"; exit 92; }
for b in 01:00.0 02:00.0 04:00.0; do
  [[ $(cat /sys/bus/pci/devices/0000:$b/current_link_speed) == '5.0 GT/s PCIe' ]] || { log "hard_stop link_speed $b"; exit 92; }
  [[ $(cat /sys/bus/pci/devices/0000:$b/current_link_width) == 8 ]] || { log "hard_stop link_width $b"; exit 92; }
done
sudo -n systemctl stop "$SIBLINGS" 2>/dev/null || true
sudo -n systemctl stop "$SERVING_UNIT"
until ! ss -tln | grep -q ':8081 '; do sleep 1; done
sudo -n systemctl reset-failed llama-research.service 2>/dev/null || true
log "variant=$LABEL launch cmd=$*"
sudo -n systemd-run --unit=llama-research.service --service-type=simple --uid=fastchip --gid=fastchip \
  --property=Restart=no --property=TimeoutStopSec=20 --property=KillMode=mixed \
  --setenv=CUDA_VISIBLE_DEVICES="$DEVICES" --setenv=GGML_CUDA_GRAPH_OPT=1 \
  --setenv=NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0} --setenv=NCCL_P2P_LEVEL=PHB \
  --setenv=LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/home/fastchip/llama.cpp-vbbb-b7d4d85-sm75/bin:/home/fastchip/llama.cpp-vbbb-b7d4d85-sm75/lib:/usr/local/cuda-12.8/lib64} \
  "$@"
"$H/wait_health.sh" http://127.0.0.1:8081/health "${HEALTH_TIMEOUT:-600}" llama-research.service
log "variant=$LABEL health_PASS"
if [[ -n ${BENCH_CMD:-} ]]; then bash -lc "$BENCH_CMD"; fi
log "variant=$LABEL bench_PASS"
