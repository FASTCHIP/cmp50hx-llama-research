#!/usr/bin/env bash
# MTP A/B on the production shape:
#   ctl (prod form, draft file idle) / no-draft / MTP n1 / MTP n3 / MTP on GPU3 / closing control.
# Config-only arms: no rebuild, one binary, the spec flags are the variable.
set -uo pipefail
ROOT=/home/fastchip/cmp50hx-llama-research
H=$ROOT/harness
RAW=$ROOT/raw
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$ROOT/logs/mtp-ab.log
PHASE=$ROOT/CURRENT_PHASE
JSONL=$RAW/mtp-ab-$STAMP.jsonl
CONC=$RAW/mtp-ab-conc-$STAMP.jsonl
PKG=/home/fastchip/llama.cpp-vbbb-mmq-thresh-sm75
DRAFT=/mnt/usbsata/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
CACHE=/mnt/usbsata/models/kvcache-ab

msk(){ TZ=Europe/Moscow date --iso-8601=seconds; }
log(){ printf '%s %s\n' "$(msk)" "$*" >>"$LOG"; }
phase(){ printf '%s' "$1" >"$PHASE"; log "phase=$1"; }

common=(--model /mnt/usbsata/models/Qwen3.8-27B-UD-Q4_K_XL.gguf --alias Qwen3.8-27B
  --mmproj /mnt/usbsata/models/mmproj-Qwen3.8-27B-Q8_0.gguf --mmproj-offload
  --host 0.0.0.0 --port 8081 --parallel 5 --ctx-size 327680 --kv-unified --kv-unified-per-slot 65536
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0.0 --repeat-penalty 1.0
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 16384 --gpu-layers 99
  --split-mode layer --tensor-split 1,1,1 --reasoning-effort low --cache-disk "$CACHE"
  --cache-disk-max 102400 -t 6 -tb 6 -b 2048 -ub 512 --api-key-file /etc/llama-server.api-key)

arm(){ # arm LABEL KIND extra...
  local label=$1 kind=$2; shift 2
  phase "$label"
  local lpath="$PKG/bin:$PKG/lib:/usr/local/cuda-12.8/lib64"
  local benchcmd="$H/mmq_bench.sh $label $JSONL $CONC $kind"
  if env LD_LIBRARY_PATH="$lpath" BENCH_CMD="$benchcmd" HEALTH_TIMEOUT=900 \
      GGML_CUDA_TURING_CUBLAS_MIN_M=256 \
      "$H/run_variant.sh" "$label" -- env GGML_CUDA_TURING_CUBLAS_MIN_M=256 "$PKG/bin/llama-server" "${common[@]}" "$@"; then
    log "arm=$label PASS"
  else
    log "arm=$label recoverable_FAIL"
  fi
  if grep -q "variant=$label NEW_XID_DETECTED" "$LOG"; then log "arm=$label ABORT_CAMPAIGN"; phase aborted-new-xid; exit 93; fi
}

log "=== mtp campaign start stamp=$STAMP ==="
arm mtp-ctl     mid   --spec-type none --model-draft "$DRAFT"
arm mtp-nodraft mid   --spec-type none
arm mtp-n3      mid   --spec-type draft-mtp --spec-draft-n-max 3
arm mtp-n1      light --spec-type draft-mtp --spec-draft-n-max 1
arm mtp-gpu3    mid   --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-device CUDA3 --override-tensor 'blk\.64\.=CUDA3'
arm mtp-ctl2    light --spec-type none --model-draft "$DRAFT"
phase done
log "=== mtp campaign complete ==="
