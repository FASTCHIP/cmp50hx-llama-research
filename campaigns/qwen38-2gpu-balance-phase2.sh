#!/usr/bin/env bash
# Phase 2: ubatch sweep on the selected two-GPU layer split.
set -uo pipefail
ROOT=/home/fastchip/cmp50hx-llama-research
H=$ROOT/harness
OUT=${OUT:?set OUT to phase-1 workspace}
SPLIT=${SPLIT:?set selected tensor split, e.g. 1.15,0.85}
RAW=$OUT/raw/ubatch-screen.jsonl
LOG=$OUT/campaign.log
PHASE=$OUT/CURRENT_PHASE
PKG=/home/fastchip/llama.cpp-upstream-mmq-a02c7f5
MODEL=/mnt/usbsata/models/Qwen3.8-27B-UD-Q4_K_XL.gguf
MMPROJ=/mnt/usbsata/models/mmproj-Qwen3.8-27B-Q8_0.gguf
mkdir -p "$OUT"/{raw,logs,state,telemetry}
exec 9>>"$OUT/logs/trace-phase2.log"; BASH_XTRACEFD=9; set -x
msk(){ TZ=Europe/Moscow date --iso-8601=seconds; }
log(){ printf '%s %s\n' "$(msk)" "$*" | tee -a "$LOG"; }
phase(){ printf '%s\n' "$1" >"$PHASE"; log "phase=$1"; }
exec 8>"$OUT/campaign.lock"
flock -n 8 || { log "hard_stop another campaign owns lock"; exit 99; }

common=(--model "$MODEL" --alias Qwen3.8-27B --mmproj "$MMPROJ" --mmproj-offload
  --host 0.0.0.0 --port 8081 --parallel 2 --ctx-size 262144 --kv-unified --kv-unified-per-slot 262144
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0.0 --repeat-penalty 1.0
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 16384 --gpu-layers 99
  --split-mode layer --tensor-split "$SPLIT" --reasoning-effort low --spec-type draft-mtp --spec-draft-n-max 3
  -t 6 -tb 6 -b 2048 --api-key-file /etc/llama-server.api-key)

arm(){
  local label=$1 ub=$2
  phase "$label"
  local arm_log="$OUT/logs/$label.log"
  local tele="$OUT/telemetry/$label-idle.csv"
  local bench="nvidia-smi --query-gpu=index,memory.used,memory.free,power.draw,clocks.current.graphics,clocks.current.memory --format=csv,noheader,nounits >'$tele' && python3 '$H/bench_request.py' --label '$label' --case short --runs 2 --classes prose,code,architecture --output '$RAW' --base http://127.0.0.1:8081 --warmup >>'$arm_log' 2>&1 && python3 '$H/bench_request.py' --label '$label' --case p4k --runs 2 --classes prose,code,architecture --output '$RAW' --base http://127.0.0.1:8081 >>'$arm_log' 2>&1 && python3 '$H/bench_request.py' --label '$label' --case p32k --runs 2 --classes prose,code,architecture --output '$RAW' --base http://127.0.0.1:8081 >>'$arm_log' 2>&1"
  log "arm=$label split=$SPLIT ubatch=$ub begin"
  if env CUDA_VISIBLE_DEVICES=0,1 \
      LD_LIBRARY_PATH="$PKG/bin:$PKG/lib:/usr/local/cuda-12.8/lib64" \
      GGML_CUDA_TURING_CUBLAS_MIN_M=256 BENCH_CMD="$bench" HEALTH_TIMEOUT=600 \
      "$H/run_variant.sh" "$label" -- "$PKG/bin/llama-server" "${common[@]}" -ub "$ub"; then
    log "arm=$label PASS"
  else
    log "arm=$label recoverable_FAIL"
  fi
  if grep -q "variant=$label NEW_XID_DETECTED" "$ROOT/logs/campaign.log"; then phase aborted-new-xid; exit 93; fi
}

log "phase2_start out=$OUT split=$SPLIT"
arm ub512 512
arm ub256 256
arm ub320 320
arm ub384 384
phase ubatch-screen-done
log "phase2_complete rows=$(wc -l <"$RAW" 2>/dev/null || echo 0)"
