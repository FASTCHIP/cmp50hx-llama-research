#!/usr/bin/env bash
# Wait for phase 3 to release the chain lock, then launch phase 4 durably.
set -uo pipefail
ROOT=/home/fastchip/bench/upstream-a02c7f5-speed
L=$ROOT/logs/chainlaunch4.log
echo "$(date -u +%FT%TZ) waiter4 start phase=$(cat $ROOT/CURRENT_PHASE 2>/dev/null)" >>"$L"
for _ in $(seq 1 700); do
  p=$(cat "$ROOT/CURRENT_PHASE" 2>/dev/null || echo none)
  if [[ $p == phase3_done ]]; then break; fi
  sleep 20
done
if [[ $(cat "$ROOT/CURRENT_PHASE" 2>/dev/null) != phase3_done ]]; then
  echo "$(date -u +%FT%TZ) phase3_never_finished phase=$(cat $ROOT/CURRENT_PHASE)" >>"$L"; exit 1
fi
echo "$(date -u +%FT%TZ) phase3 done; launching phase4" >>"$L"
sudo -n systemd-run --unit=bench-phase4 --collect --property=Type=oneshot --no-block \
  --setenv=NO_PROXY='*' \
  /bin/bash -lc "$ROOT/harness/chain_phase4.sh 20260924T2050Z layer-ub2048-clean ub2048-vision2 base-vision tensor-ub2048 tensor-ub1024 b4096-p32k" \
  >>"$L" 2>&1
echo "$(date -u +%FT%TZ) phase4_launch_rc=$?" >>"$L"
