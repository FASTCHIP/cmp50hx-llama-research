#!/usr/bin/env bash
# MTP draft-length sweep: find the peak beyond n-max 3 (light suites, config-only).
# Queues itself behind any running campaign (single-tenant stand).
set -uo pipefail
ROOT=/home/fastchip/cmp50hx-llama-research
H=$ROOT/harness
RAW=$ROOT/raw
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$ROOT/logs/mtp-nmax.log
PHASE=$ROOT/CURRENT_PHASE
JSONL=$RAW/mtp-ab-$STAMP.jsonl
CONC=$RAW/mtp-ab-conc-$STAMP.jsonl
PKG=/home/fastchip/llama.cpp-vbbb-mmq-thresh-sm75
CACHE=/mnt/usbsata/models/kvcache-ab

msk(){ TZ=Europe/Moscow date --iso-8601=seconds; }
log(){ printf '%s %s\n' "$(msk)" "$*" >>"$LOG"; }
phase(){ printf '%s' "$1" >"$PHASE"; log "phase=$1"; }

for i in $(seq 1 90); do
  ph=$(cat "$PHASE" 2>/dev/null)
  case "$ph" in done|aborted-new-xid) break;; esac
  sleep 60
done
log "=== nmax sweep start stamp=$STAMP (previous phase=$ph) ==="

common=(--model /mnt/usbsata/models/Qwen3.8-27B-UD-Q4_K_XL.gguf --alias Qwen3.8-27B
  --mmproj /mnt/usbsata/models/mmproj-Qwen3.8-27B-Q8_0.gguf --mmproj-offload
  --host 0.0.0.0 --port 8081 --parallel 5 --ctx-size 327680 --kv-unified --kv-unified-per-slot 65536
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0.0 --repeat-penalty 1.0
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 16384 --gpu-layers 99
  --split-mode layer --tensor-split 1,1,1 --reasoning-effort low --cache-disk "$CACHE"
  --cache-disk-max 102400 -t 6 -tb 6 -b 2048 -ub 512 --api-key-file /etc/llama-server.api-key)

arm(){ local label=$1 kind=$2; shift 2
  phase "$label"
  local benchcmd="$H/mmq_bench.sh $label $JSONL $CONC $kind"
  if env LD_LIBRARY_PATH="$PKG/bin:$PKG/lib:/usr/local/cuda-12.8/lib64" BENCH_CMD="$benchcmd" HEALTH_TIMEOUT=900 \
      "$H/run_variant.sh" "$label" -- env GGML_CUDA_TURING_CUBLAS_MIN_M=256 "$PKG/bin/llama-server" "${common[@]}" "$@"; then
    log "arm=$label PASS"
  else
    log "arm=$label recoverable_FAIL"
  fi
  if grep -q "variant=$label NEW_XID_DETECTED" "$LOG"; then log "arm=$label ABORT"; phase aborted-new-xid; exit 93; fi
}

arm mtp-n5 light --spec-type draft-mtp --spec-draft-n-max 5
arm mtp-n7 light --spec-type draft-mtp --spec-draft-n-max 7
arm mtp-n3b light --spec-type draft-mtp --spec-draft-n-max 3
phase done
log "=== nmax sweep complete ==="
