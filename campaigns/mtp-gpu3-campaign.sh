#!/usr/bin/env bash
# Remaining MTP arms: the GPU3 offload (fixed device visibility) and the closing control.
# The earlier attempt failed with 'invalid device: CUDA3' because the runner exports
# CUDA_VISIBLE_DEVICES=0,1,2 by default, so GPU3 does not exist for the process.
set -uo pipefail
ROOT=/home/fastchip/cmp50hx-llama-research
H=$ROOT/harness
RAW=$ROOT/raw
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$ROOT/logs/mtp-gpu3.log
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

arm(){ # arm LABEL KIND DEVICES extra...
  local label=$1 kind=$2 devices=$3; shift 3
  phase "$label"
  local lpath="$PKG/bin:$PKG/lib:/usr/local/cuda-12.8/lib64"
  local benchcmd="$H/mmq_bench.sh $label $JSONL $CONC $kind"
  if env CUDA_VISIBLE_DEVICES="$devices" LD_LIBRARY_PATH="$lpath" BENCH_CMD="$benchcmd" HEALTH_TIMEOUT=900 \
      "$H/run_variant.sh" "$label" -- env GGML_CUDA_TURING_CUBLAS_MIN_M=256 "$PKG/bin/llama-server" "${common[@]}" "$@"; then
    log "arm=$label PASS"
  else
    log "arm=$label recoverable_FAIL"
  fi
  if grep -q "variant=$label NEW_XID_DETECTED" "$LOG"; then log "arm=$label ABORT"; phase aborted-new-xid; exit 93; fi
}

log "=== mtp gpu3 campaign start stamp=$STAMP ==="
# model on the three 20 GiB cards, MTP block and draft context on the 10 GiB card (Gen2 x4)
arm mtp-gpu3 mid 0,1,2,3 --device CUDA0,CUDA1,CUDA2 --spec-type draft-mtp --spec-draft-n-max 3 \
    --spec-draft-device CUDA3 --override-tensor 'blk\.64\.=CUDA3'
arm mtp-ctl2 light 0,1,2 --spec-type none --model-draft "$DRAFT"
phase done
log "=== mtp gpu3 campaign complete ==="
