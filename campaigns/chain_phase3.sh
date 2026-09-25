#!/usr/bin/env bash
# Phase 3: long-context reference arm, 5-rep confirmation of the ubatch winner,
# the mtp5 prose anomaly re-check, and the context-ceiling gate.
set -uo pipefail
ROOT=/home/fastchip/bench/upstream-a02c7f5-speed
H=$ROOT/harness
STAMP=${1:?usage: chain_phase3.sh STAMP ARM...}; shift
LOG=$ROOT/logs/phase3-$STAMP.log
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

BASE=( "$BIN"
  --model "$MODEL" --alias Qwen3.8-27B
  --mmproj "$MMP" --mmproj-offload
  --host 0.0.0.0 --port 8081
  --parallel 4 --ctx-size 524288 --kv-unified --kv-unified-per-slot 262144
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0.0 --repeat-penalty 1.0
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0
  --gpu-layers 99 --split-mode layer --tensor-split 1.2,1.2,0.6
  --reasoning-effort low -t 6 -tb 6 -b 2048
  --api-key-file /etc/llama-server.api-key )

spec(){
  case $1 in
    base-long)    MODE=direct EXTRA="" ENVV=""
                  BENCH="$H/run_arm.sh $STAMP base-long 'p64k:1 p120k:1 p192k:1'" ;;
    ub2048-rep5)  MODE=variant EXTRA="--cache-ram 16384 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 3" ENVV="GGML_CUDA_TURING_CUBLAS_MIN_M=256"
                  BENCH="$H/run_arm.sh $STAMP ub2048-rep5 'p32k:5 short:5'" ;;
    ub1024-rep5)  MODE=variant EXTRA="--cache-ram 16384 -ub 1024 --spec-type draft-mtp --spec-draft-n-max 3" ENVV="GGML_CUDA_TURING_CUBLAS_MIN_M=256"
                  BENCH="$H/run_arm.sh $STAMP ub1024-rep5 'p32k:5 short:5'" ;;
    mtp5-p4k)     MODE=variant EXTRA="--cache-ram 16384 -ub 512 --spec-type draft-mtp --spec-draft-n-max 5" ENVV="GGML_CUDA_TURING_CUBLAS_MIN_M=256"
                  BENCH="$H/run_arm.sh $STAMP mtp5-p4k 'p4k:3'" ;;
    ub2048-p250k) MODE=variant EXTRA="--cache-ram 16384 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 3" ENVV="GGML_CUDA_TURING_CUBLAS_MIN_M=256"
                  BENCH="$H/run_arm.sh $STAMP ub2048-p250k 'p250k:1'" ;;
    ub2048-vision) MODE=variant EXTRA="--cache-ram 16384 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 3" ENVV="GGML_CUDA_TURING_CUBLAS_MIN_M=256"
                  BENCH="$H/vision_smoke.py > $ROOT/baseline/vision-p3-$STAMP.json; $H/run_arm.sh $STAMP ub2048-vision 'short:2'" ;;
    *) MODE=none EXTRA="" ENVV="" BENCH="" ;;
  esac
}

log "phase3_begin stamp=$STAMP arms='$*'"
for a in "$@"; do
  printf '%s' "$a" >"$ROOT/CURRENT_PHASE"
  spec "$a"
  case $MODE in
    none) log "arm=$a UNKNOWN_ARM"; continue ;;
    direct) rc=0; bash -lc "$BENCH" || rc=$? ;;
    variant)
      read -r -a EXTRA_A <<<"$EXTRA"; read -r -a ENVV_A <<<"$ENVV"
      rc=0
      BENCH_CMD="$BENCH" env "${ENVV_A[@]}" "$H/run_variant.sh" "$a" -- "${BASE[@]}" "${EXTRA_A[@]}" >>"$LOG" 2>&1 || rc=$? ;;
  esac
  log "arm=$a end rc=$rc"
  if (( rc >= 90 )); then log "HARD_STOP arm=$a rc=$rc"; printf 'hard_stop-%s' "$a" >"$ROOT/CURRENT_PHASE"; break; fi
done
printf 'phase3_done' >"$ROOT/CURRENT_PHASE"
log "phase3_complete arms='$*'"
