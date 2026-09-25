#!/usr/bin/env bash
# Phase 4: tensor-split combined with the ubatch winner, clean re-run of the arms
# that caught a degraded server state, and a vision gate on both reference and winner.
set -uo pipefail
ROOT=/home/fastchip/bench/upstream-a02c7f5-speed
H=$ROOT/harness
STAMP=${1:?usage: chain_phase4.sh STAMP ARM...}; shift
LOG=$ROOT/logs/phase4-$STAMP.log
PKG=/home/fastchip/llama.cpp-upstream-mmq-a02c7f5
BIN=$PKG/bin/llama-server
MODEL=/mnt/usbsata/models/Qwen3.8-27B-UD-Q4_K_XL.gguf
MMP=/mnt/usbsata/models/mmproj-Qwen3.8-27B-Q8_0.gguf

export LD_LIBRARY_PATH=$PKG/bin:$PKG/lib:/usr/local/cuda-12.8/lib64
export CUDA_VISIBLE_DEVICES=0,1,2
export GGML_CUDA_GRAPH_OPT=1 NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=PHB GGML_CUDA_TURING_CUBLAS_MIN_M=256
export NO_PROXY='*'

exec 200>"$ROOT/logs/chain.lock"
flock -n 200 || { echo "another chain holds the lock" >&2; exit 99; }

log(){ printf '%s %s\n' "$(TZ=Europe/Moscow date --iso-8601=seconds)" "$*" >>"$LOG"; }
mkdir -p "$ROOT/baseline" "$ROOT/raw" "$ROOT/summaries"

# common profile WITHOUT the split flags: phase 4 passes them once, per arm
BASE=( "$BIN"
  --model "$MODEL" --alias Qwen3.8-27B
  --mmproj "$MMP" --mmproj-offload
  --host 0.0.0.0 --port 8081
  --parallel 4 --ctx-size 524288 --kv-unified --kv-unified-per-slot 262144
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0.0 --repeat-penalty 1.0
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0
  --gpu-layers 99 --reasoning-effort low -t 6 -tb 6 -b 2048
  --api-key-file /etc/llama-server.api-key )

spec(){
  case $1 in
    tensor-ub2048)  MODE=variant
      EXTRA="--cache-ram 16384 -ub 2048 -b 2048 --spec-type draft-mtp --spec-draft-n-max 3 --split-mode tensor --tensor-split 1,1,1"
      BENCH="$H/run_arm.sh $STAMP tensor-ub2048 'short:3 p4k:3 p32k:3'" ;;
    tensor-ub1024)  MODE=variant
      EXTRA="--cache-ram 16384 -ub 1024 --spec-type draft-mtp --spec-draft-n-max 3 --split-mode tensor --tensor-split 1,1,1"
      BENCH="$H/run_arm.sh $STAMP tensor-ub1024 'short:3 p4k:3 p32k:3'" ;;
    tensor-ub2048-long) MODE=variant
      EXTRA="--cache-ram 16384 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 3 --split-mode tensor --tensor-split 1,1,1"
      BENCH="$H/run_arm.sh $STAMP tensor-ub2048-long 'p64k:1 p120k:1 p192k:1'" ;;
    layer-ub2048-clean) MODE=variant
      EXTRA="--cache-ram 16384 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 3 --split-mode layer --tensor-split 1.2,1.2,0.6"
      BENCH="$H/run_arm.sh $STAMP layer-ub2048-clean 'short:3 p4k:3 p32k:3'" ;;
    base-vision)    MODE=direct
      BENCH="$H/vision_smoke.py > $ROOT/baseline/vision-base-$STAMP.json; $H/run_arm.sh $STAMP base-vision 'short:2'" ;;
    ub2048-vision2) MODE=variant
      EXTRA="--cache-ram 16384 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 3 --split-mode layer --tensor-split 1.2,1.2,0.6"
      BENCH="$H/vision_smoke.py > $ROOT/baseline/vision-ub2048v2-$STAMP.json; $H/run_arm.sh $STAMP ub2048-vision2 'short:2'" ;;
    b4096-p32k)     MODE=variant
      EXTRA="--cache-ram 16384 -b 4096 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 3 --split-mode layer --tensor-split 1.2,1.2,0.6"
      BENCH="$H/run_arm.sh $STAMP b4096-p32k 'p32k:2'" ;;
    *) MODE=none ;;
  esac
}

log "phase4_begin stamp=$STAMP arms='$*'"
for a in "$@"; do
  printf '%s' "$a" >"$ROOT/CURRENT_PHASE"
  spec "$a"
  case $MODE in
    none) log "arm=$a UNKNOWN_ARM"; continue ;;
    direct) rc=0; bash -lc "$BENCH" || rc=$? ;;
    variant)
      read -r -a EXTRA_A <<<"$EXTRA"
      rc=0
      BENCH_CMD="$BENCH" GGML_CUDA_TURING_CUBLAS_MIN_M=256 "$H/run_variant.sh" "$a" -- "${BASE[@]}" "${EXTRA_A[@]}" >>"$LOG" 2>&1 || rc=$? ;;
  esac
  log "arm=$a end rc=$rc"
  if (( rc >= 90 )); then log "HARD_STOP arm=$a rc=$rc"; printf 'hard_stop-%s' "$a" >"$ROOT/CURRENT_PHASE"; break; fi
done
printf 'phase4_done' >"$ROOT/CURRENT_PHASE"
log "phase4_complete arms='$*'"
