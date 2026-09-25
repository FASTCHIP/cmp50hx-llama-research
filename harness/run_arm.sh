#!/usr/bin/env bash
# Baseline / screening arm: benchmarks the service that is ALREADY running on :8081.
# Never stops or restarts anything. Reads the API only.
set -uo pipefail
ROOT=/home/fastchip/bench/upstream-a02c7f5-speed
H=$ROOT/harness
STAMP=${1:?usage: run_arm.sh STAMP LABEL [CASES]}
LABEL=${2:?label}
CASES=${3:-"short:3 p4k:3 p32k:3"}
BASE=${BASE:-http://127.0.0.1:8081}
OUT=$ROOT/raw/$LABEL-$STAMP.jsonl
LOG=$ROOT/logs/$LABEL-$STAMP.log
: >"$LOG"
echo "[$(date -u +%FT%TZ)] arm=$LABEL cases='$CASES' base=$BASE out=$OUT" >>"$LOG"
rc_all=0
for c in $CASES; do
  case=${c%%:*}; runs=${c##*:}
  echo "[$(date -u +%FT%TZ)] case=$case runs=$runs begin" >>"$LOG"
  "$H/bench_request.py" --label "$LABEL" --case "$case" --runs "$runs" --output "$OUT" --base "$BASE" --warmup >>"$LOG" 2>&1
  rc=$?
  echo "[$(date -u +%FT%TZ)] case=$case end rc=$rc rows=$(wc -l <\"$OUT\" 2>/dev/null || echo 0)" >>"$LOG"
  [[ $rc -ne 0 ]] && rc_all=$rc
done
echo "[$(date -u +%FT%TZ)] arm_complete label=$LABEL rows=$(wc -l <"$OUT") rc=$rc_all" >>"$LOG"
exit $rc_all
