#!/usr/bin/env bash
# Bench suites for the MMQ threshold A/B. Mirrors the campaign vocabulary
# (short / p4k / p32k / p64k + concurrency 1,2,5) so rows stay comparable.
set -uo pipefail
H=/home/fastchip/cmp50hx-llama-research/harness
LABEL=${1:?label}; JSONL=${2:?jsonl}; CONC=${3:?conc}; KIND=${4:-full}
CLASSES=prose,code,architecture

if [[ $KIND == full ]]; then
  python3 "$H/bench_request.py" --label "$LABEL" --case short --runs 2 --classes "$CLASSES" --output "$JSONL" --warmup
  python3 "$H/bench_request.py" --label "$LABEL" --case p4k   --runs 2 --classes "$CLASSES" --output "$JSONL"
  python3 "$H/bench_request.py" --label "$LABEL" --case p32k  --runs 2 --classes "$CLASSES" --output "$JSONL"
  python3 "$H/bench_request.py" --label "$LABEL" --case p64k  --runs 1 --classes "$CLASSES" --output "$JSONL"
  for n in 1 2 5; do python3 "$H/bench_concurrency.py" --label "$LABEL" --concurrency "$n" --runs 2 --output "$CONC"; done
else
  python3 "$H/bench_request.py" --label "$LABEL" --case short --runs 2 --classes prose,code --output "$JSONL" --warmup
  python3 "$H/bench_request.py" --label "$LABEL" --case p32k  --runs 2 --classes prose,code --output "$JSONL"
  python3 "$H/bench_concurrency.py" --label "$LABEL" --concurrency 5 --runs 2 --output "$CONC"
fi
echo "bench suite done: $LABEL ($KIND)"
