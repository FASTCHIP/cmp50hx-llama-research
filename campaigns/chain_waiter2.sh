#!/usr/bin/env bash
# Wait for phase 1 to release the chain lock, then launch phase 2 durably.
set -uo pipefail
ROOT=/home/fastchip/bench/upstream-a02c7f5-speed
L=$ROOT/logs/chainlaunch2.log
echo "$(date -u +%FT%TZ) waiter2 start phase=$(cat $ROOT/CURRENT_PHASE 2>/dev/null)" >>"$L"
for _ in $(seq 1 240); do
  p=$(cat "$ROOT/CURRENT_PHASE" 2>/dev/null || echo none)
  if [[ $p == chain_done ]]; then break; fi
  sleep 15
done
if [[ $(cat "$ROOT/CURRENT_PHASE" 2>/dev/null) != chain_done ]]; then
  echo "$(date -u +%FT%TZ) phase1_never_finished phase=$(cat $ROOT/CURRENT_PHASE)" >>"$L"; exit 1
fi
echo "$(date -u +%FT%TZ) phase1 done; launching phase2" >>"$L"
sudo -n systemd-run --unit=bench-phase2 --collect --property=Type=oneshot --no-block \
  --setenv=NO_PROXY='*' \
  /bin/bash -lc "$ROOT/harness/chain_phase2.sh 20260924T2050Z ub2048-long ub1024-long base-conc ub2048-conc tensor-mtp3 tensor-long ub2048-vision ub1024-conc ub2048-b4096 ub2048-tb12" \
  >>"$L" 2>&1
echo "$(date -u +%FT%TZ) phase2_launch_rc=$?" >>"$L"
