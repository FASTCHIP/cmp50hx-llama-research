#!/usr/bin/env bash
# Wait for phase 2 to release the chain lock, then launch phase 3 durably.
set -uo pipefail
ROOT=/home/fastchip/bench/upstream-a02c7f5-speed
L=$ROOT/logs/chainlaunch3.log
echo "$(date -u +%FT%TZ) waiter3 start phase=$(cat $ROOT/CURRENT_PHASE 2>/dev/null)" >>"$L"
for _ in $(seq 1 400); do
  p=$(cat "$ROOT/CURRENT_PHASE" 2>/dev/null || echo none)
  if [[ $p == phase2_done ]]; then break; fi
  sleep 15
done
if [[ $(cat "$ROOT/CURRENT_PHASE" 2>/dev/null) != phase2_done ]]; then
  echo "$(date -u +%FT%TZ) phase2_never_finished phase=$(cat $ROOT/CURRENT_PHASE)" >>"$L"; exit 1
fi
echo "$(date -u +%FT%TZ) phase2 done; launching phase3" >>"$L"
sudo -n systemd-run --unit=bench-phase3 --collect --property=Type=oneshot --no-block \
  --setenv=NO_PROXY='*' \
  /bin/bash -lc "$ROOT/harness/chain_phase3.sh 20260924T2050Z base-long ub2048-rep5 ub1024-rep5 mtp5-p4k ub2048-p250k" \
  >>"$L" 2>&1
echo "$(date -u +%FT%TZ) phase3_launch_rc=$?" >>"$L"
