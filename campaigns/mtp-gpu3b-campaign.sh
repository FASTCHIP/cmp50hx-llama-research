#!/usr/bin/env bash
# Fixed GPU3 arm: MTP block and draft context on the 10 GiB card, model on GPU0-2.
# Fixes the two failed attempts:
#   1) CUDA_VISIBLE_DEVICES must include the card (else 'invalid device: CUDA3');
#   2) no --device restriction (it drops CUDA3's buffer type -> 'cannot run the operation (NONE)');
#      instead all four devices stay visible and the model gets a zero share on GPU3.
# Waits for any running campaign to finish first (single-tenant stand).
set -uo pipefail
ROOT=/home/fastchip/cmp50hx-llama-research
H=$ROOT/harness
RAW=$ROOT/raw
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$ROOT/logs/mtp-gpu3b.log
PHASE=$ROOT/CURRENT_PHASE
JSONL=$RAW/mtp-ab-$STAMP.jsonl
CONC=$RAW/mtp-ab-conc-$STAMP.jsonl
PKG=/home/fastchip/llama.cpp-vbbb-mmq-thresh-sm75
DRAFT=/mnt/usbsata/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
CACHE=/mnt/usbsata/models/kvcache-ab

msk(){ TZ=Europe/Moscow date --iso-8601=seconds; }
log(){ printf '%s %s\n' "$(msk)" "$*" >>"$LOG"; }
phase(){ printf '%s' "$1" >"$PHASE"; log "phase=$1"; }

# wait for the previous campaign to release the stand
for i in $(seq 1 90); do
  ph=$(cat "$PHASE" 2>/dev/null)
  case "$ph" in done|aborted-new-xid) break;; esac
  sleep 60
done
log "=== gpu3-fixed campaign start stamp=$STAMP (previous phase=$ph) ==="

common=(--model /mnt/usbsata/models/Qwen3.8-27B-UD-Q4_K_XL.gguf --alias Qwen3.8-27B
  --mmproj /mnt/usbsata/models/mmproj-Qwen3.8-27B-Q8_0.gguf --mmproj-offload
  --host 0.0.0.0 --port 8081 --parallel 5 --ctx-size 327680 --kv-unified --kv-unified-per-slot 65536
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0.0 --repeat-penalty 1.0
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 16384 --gpu-layers 99
  --split-mode layer --tensor-split 1,1,1,0 --reasoning-effort low --cache-disk "$CACHE"
  --cache-disk-max 102400 -t 6 -tb 6 -b 2048 -ub 512 --api-key-file /etc/llama-server.api-key)

arm(){ # arm LABEL KIND extra...
  local label=$1 kind=$2; shift 2
  phase "$label"
  local lpath="$PKG/bin:$PKG/lib:/usr/local/cuda-12.8/lib64"
  local benchcmd="$H/mmq_bench.sh $label $JSONL $CONC $kind"
  if env CUDA_VISIBLE_DEVICES=0,1,2,3 LD_LIBRARY_PATH="$lpath" BENCH_CMD="$benchcmd" HEALTH_TIMEOUT=900 \
      "$H/run_variant.sh" "$label" -- env GGML_CUDA_TURING_CUBLAS_MIN_M=256 "$PKG/bin/llama-server" "${common[@]}" "$@"; then
    log "arm=$label PASS"
  else
    log "arm=$label recoverable_FAIL"
  fi
  if grep -q "variant=$label NEW_XID_DETECTED" "$LOG"; then log "arm=$label ABORT"; phase aborted-new-xid; exit 93; fi
}

arm mtp-gpu3b mid --spec-type draft-mtp --spec-draft-n-max 3 \
    --spec-draft-device CUDA3 --override-tensor 'blk\.64\.=CUDA3'
arm mtp-gpu3b2 light --spec-type draft-mtp --spec-draft-n-max 3 \
    --spec-draft-device CUDA3 --override-tensor 'blk\.64\.=CUDA3'
phase done
log "=== gpu3-fixed campaign complete ==="
