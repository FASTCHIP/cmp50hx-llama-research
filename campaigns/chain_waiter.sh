#!/usr/bin/env bash
# Wait for the baseline arm to finish, then hand off to the durable screening chain.
set -uo pipefail
ROOT=/home/fastchip/bench/upstream-a02c7f5-speed
L=$ROOT/logs/chainlaunch.log
BLOG=$ROOT/logs/base-a02c7f5-20260924T2050Z.log
rows_now(){ wc -l <"$ROOT/raw/base-a02c7f5-20260924T2050Z.jsonl" 2>/dev/null || echo 0; }
echo "$(date -u +%FT%TZ) waiter start rows=$(rows_now)" >>"$L"
for _ in $(seq 1 200); do
  if grep -q arm_complete "$BLOG" 2>/dev/null; then break; fi
  sleep 15
done
if ! grep -q arm_complete "$BLOG" 2>/dev/null; then
  echo "$(date -u +%FT%TZ) baseline_never_completed rows=$(rows_now)" >>"$L"
  exit 1
fi
rows=$(rows_now)
echo "$(date -u +%FT%TZ) baseline_complete rows=$rows; starting chain" >>"$L"
if (( rows < 20 )); then
  echo "$(date -u +%FT%TZ) too_few_baseline_rows=$rows; not starting chain" >>"$L"
  exit 2
fi
sudo -n systemd-run --unit=bench-chain --collect --property=Type=oneshot --no-block \
  --setenv=NO_PROXY='*' \
  /bin/bash -lc "$ROOT/harness/chain_screen.sh 20260924T2050Z ub1024 ub2048 base2 mtp5 mtp1 nospec thr0 thr512 cram4096 base3" \
  >>"$L" 2>&1
echo "$(date -u +%FT%TZ) chain_launch_rc=$?" >>"$L"
