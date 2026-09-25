#!/usr/bin/env bash
# Durable screening chain for the upstream-a02c7f5-speed campaign.
# One arm at a time; each arm stops the serving unit, launches the variant on :8081,
# health-gates, benchmarks, and restores the serving unit through run_variant.sh's trap.
# Usage: chain_screen.sh STAMP arm1 arm2 ...
set -uo pipefail
ROOT=/home/fastchip/bench/upstream-a02c7f5-speed
H=$ROOT/harness
STAMP=${1:?usage: chain_screen.sh STAMP ARM...}; shift
LOG=$ROOT/logs/chain-$STAMP.log
PKG=/home/fastchip/llama.cpp-upstream-mmq-a02c7f5
BIN=$PKG/bin/llama-server
MODEL=/mnt/usbsata/models/Qwen3.8-27B-UD-Q4_K_XL.gguf
MMP=/mnt/usbsata/models/mmproj-Qwen3.8-27B-Q8_0.gguf
CASES=${CASES:-"short:3 p4k:3 p32k:3"}

export LD_LIBRARY_PATH=$PKG/bin:$PKG/lib:/usr/local/cuda-12.8/lib64
export CUDA_VISIBLE_DEVICES=0,1,2
export GGML_CUDA_GRAPH_OPT=1 NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=PHB GGML_CUDA_TURING_CUBLAS_MIN_M=256
export NO_PROXY='*'

exec 200>"$ROOT/logs/chain.lock"
flock -n 200 || { echo "another chain holds the lock" >&2; exit 99; }

log(){ printf '%s %s\n' "$(TZ=Europe/Moscow date --iso-8601=seconds)" "$*" >>"$LOG"; }

# common profile: exactly the live serving configuration, minus what each arm varies
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

arm_extra(){
  case $1 in
    base)            echo "--cache-ram 16384 -ub 512 --spec-type draft-mtp --spec-draft-n-max 3" ;;
    ub1024)          echo "--cache-ram 16384 -ub 1024 --spec-type draft-mtp --spec-draft-n-max 3" ;;
    ub2048)          echo "--cache-ram 16384 -ub 2048 --spec-type draft-mtp --spec-draft-n-max 3" ;;
    mtp5)            echo "--cache-ram 16384 -ub 512 --spec-type draft-mtp --spec-draft-n-max 5" ;;
    mtp1)            echo "--cache-ram 16384 -ub 512 --spec-type draft-mtp --spec-draft-n-max 1" ;;
    nospec)          echo "--cache-ram 16384 -ub 512 --spec-type none" ;;
    thr0)            echo "--cache-ram 16384 -ub 512 --spec-type draft-mtp --spec-draft-n-max 3" ;;
    thr512)          echo "--cache-ram 16384 -ub 512 --spec-type draft-mtp --spec-draft-n-max 3" ;;
    cram4096)        echo "--cache-ram 4096 -ub 512 --spec-type draft-mtp --spec-draft-n-max 3" ;;
    b4096)           echo "--cache-ram 16384 -b 4096 -ub 1024 --spec-type draft-mtp --spec-draft-n-max 3" ;;
    tb12)            echo "--cache-ram 16384 -ub 512 -t 6 -tb 12 --spec-type draft-mtp --spec-draft-n-max 3" ;;
    *) echo "" ;;
  esac
}
arm_env(){
  case $1 in
    thr0)   echo "GGML_CUDA_TURING_CUBLAS_MIN_M=0" ;;
    thr512) echo "GGML_CUDA_TURING_CUBLAS_MIN_M=512" ;;
    *)      echo "GGML_CUDA_TURING_CUBLAS_MIN_M=256" ;;
  esac
}

log "chain_begin stamp=$STAMP arms='$*' cases='$CASES'"
for a in "$@"; do
  printf '%s' "$a" >"$ROOT/CURRENT_PHASE"
  log "arm=$a begin"
  if [[ $a == base* ]]; then
    # drift control: bench the RESTORED SERVING unit, do not stop anything
    rc=0
    "$H/run_arm.sh" "$STAMP" "$a" "$CASES" || rc=$?
  else
    read -r -a EXTRA <<<"$(arm_extra "$a")"
    read -r -a ENVV  <<<"$(arm_env "$a")"
    [[ ${#EXTRA[@]} -gt 0 ]] || { log "arm=$a UNKNOWN_ARM"; continue; }
    rc=0
    BENCH_CMD="$H/run_arm.sh $STAMP $a '$CASES'" \
      env "${ENVV[@]}" "$H/run_variant.sh" "$a" -- "${BASE[@]}" "${EXTRA[@]}" >>"$LOG" 2>&1 || rc=$?
  fi
  log "arm=$a end rc=$rc rows=$(wc -l <"$ROOT/raw/$a-$STAMP.jsonl" 2>/dev/null || echo 0)"
  if (( rc >= 90 )); then
    log "HARD_STOP arm=$a rc=$rc"
    printf 'hard_stop-%s' "$a" >"$ROOT/CURRENT_PHASE"
    break
  fi
done
printf 'chain_done' >"$ROOT/CURRENT_PHASE"
log "chain_complete arms='$*'"
