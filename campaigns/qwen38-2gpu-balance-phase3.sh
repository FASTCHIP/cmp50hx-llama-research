#!/usr/bin/env bash
# Phase 3: compare final layer/pipeline profile with two-GPU tensor split.
set -uo pipefail
ROOT=/home/fastchip/cmp50hx-llama-research
H=$ROOT/harness
OUT=${OUT:?set OUT to campaign workspace}
SPLIT=${SPLIT:?selected layer tensor split}
UBATCH=${UBATCH:?selected ubatch}
RAW=$OUT/raw/mode-finalists.jsonl
CONC=$OUT/raw/mode-finalists-conc.jsonl
LOG=$OUT/campaign.log
PHASE=$OUT/CURRENT_PHASE
PKG=/home/fastchip/llama.cpp-upstream-mmq-a02c7f5
MODEL=/mnt/usbsata/models/Qwen3.8-27B-UD-Q4_K_XL.gguf
MMPROJ=/mnt/usbsata/models/mmproj-Qwen3.8-27B-Q8_0.gguf
mkdir -p "$OUT"/{raw,logs,state,telemetry}
exec 9>>"$OUT/logs/trace-phase3.log"; BASH_XTRACEFD=9; set -x
msk(){ TZ=Europe/Moscow date --iso-8601=seconds; }
log(){ printf '%s %s\n' "$(msk)" "$*" | tee -a "$LOG"; }
phase(){ printf '%s\n' "$1" >"$PHASE"; log "phase=$1"; }
exec 8>"$OUT/campaign.lock"
flock -n 8 || { log "hard_stop another campaign owns lock"; exit 99; }

base=(--model "$MODEL" --alias Qwen3.8-27B --mmproj "$MMPROJ" --mmproj-offload
  --host 0.0.0.0 --port 8081 --parallel 2 --ctx-size 262144 --kv-unified --kv-unified-per-slot 262144
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0.0 --repeat-penalty 1.0
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 16384 --gpu-layers 99
  --reasoning-effort low --spec-type draft-mtp --spec-draft-n-max 3
  -t 6 -tb 6 -b 2048 -ub "$UBATCH" --api-key-file /etc/llama-server.api-key)

arm(){
  local label=$1 mode=$2 split=$3
  phase "$label"
  local arm_log="$OUT/logs/$label.log"
  local tele="$OUT/telemetry/$label-idle.csv"
  local bench="nvidia-smi --query-gpu=index,memory.used,memory.free,power.draw,clocks.current.graphics,clocks.current.memory --format=csv,noheader,nounits >'$tele' && python3 '$H/bench_request.py' --label '$label' --case p32k --runs 1 --classes prose,code,architecture --output '$RAW' --base http://127.0.0.1:8081 >>'$arm_log' 2>&1 && python3 '$H/bench_request.py' --label '$label' --case p120k --runs 1 --classes prose,code,architecture --output '$RAW' --base http://127.0.0.1:8081 >>'$arm_log' 2>&1 && python3 '$H/bench_concurrency.py' --label '$label' --concurrency 2 --runs 3 --n-predict 256 --output '$CONC' --base http://127.0.0.1:8081 >>'$arm_log' 2>&1"
  log "arm=$label mode=$mode split=$split ubatch=$UBATCH begin"
  if env CUDA_VISIBLE_DEVICES=0,1 \
      LD_LIBRARY_PATH="$PKG/bin:$PKG/lib:/usr/local/cuda-12.8/lib64" \
      GGML_CUDA_TURING_CUBLAS_MIN_M=256 BENCH_CMD="$bench" HEALTH_TIMEOUT=600 \
      "$H/run_variant.sh" "$label" -- "$PKG/bin/llama-server" "${base[@]}" --split-mode "$mode" --tensor-split "$split"; then
    log "arm=$label PASS"
  else
    log "arm=$label recoverable_FAIL"
  fi
  if grep -q "variant=$label NEW_XID_DETECTED" "$ROOT/logs/campaign.log"; then phase aborted-new-xid; exit 93; fi
}

log "phase3_start out=$OUT split=$SPLIT ubatch=$UBATCH"
arm mode-layer layer "$SPLIT"
arm mode-tensor tensor 1.00,1.00
phase mode-comparison-done
log "phase3_complete rows=$(wc -l <"$RAW" 2>/dev/null || echo 0) conc=$(wc -l <"$CONC" 2>/dev/null || echo 0)"
