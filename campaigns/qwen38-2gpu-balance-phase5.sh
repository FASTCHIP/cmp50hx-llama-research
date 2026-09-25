#!/usr/bin/env bash
# Phase 5: disambiguate split vs ubatch at the 256K gate, on the balance winner.
set -uo pipefail
ROOT=/home/fastchip/cmp50hx-llama-research
H=$ROOT/harness
OUT=${OUT:?campaign workspace}
RAW=$OUT/raw/p250-disambig.jsonl
CONC=$OUT/raw/p250-disambig-conc.jsonl
LOG=$OUT/campaign.log
PHASE=$OUT/CURRENT_PHASE
PKG=/home/fastchip/llama.cpp-upstream-mmq-a02c7f5
MODEL=/mnt/usbsata/models/Qwen3.8-27B-UD-Q4_K_XL.gguf
MMPROJ=/mnt/usbsata/models/mmproj-Qwen3.8-27B-Q8_0.gguf
mkdir -p "$OUT"/{raw,logs,state,telemetry}
msk(){ TZ=Europe/Moscow date --iso-8601=seconds; }
log(){ printf '%s %s\n' "$(msk)" "$*" | tee -a "$LOG"; }
phase(){ printf '%s\n' "$1" >"$PHASE"; log "phase=$1"; }
exec 8>"$OUT/campaign.lock"; flock -n 8 || { log "hard_stop lock"; exit 99; }
base=(--model "$MODEL" --alias Qwen3.8-27B --mmproj "$MMPROJ" --mmproj-offload --host 0.0.0.0 --port 8081
 --parallel 2 --ctx-size 262144 --kv-unified --kv-unified-per-slot 262144 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0
 --presence-penalty 0.0 --repeat-penalty 1.0 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 16384
 --gpu-layers 99 --split-mode layer --tensor-split 1.10,0.90 --reasoning-effort low --spec-type draft-mtp --spec-draft-n-max 3
 -t 6 -tb 6 -b 2048 --api-key-file /etc/llama-server.api-key)
label=p250-split512
phase "$label"
alog="$OUT/logs/$label.log"
bench="python3 '$H/bench_request.py' --label '$label' --case p250k --runs 1 --classes prose --output '$RAW' --base http://127.0.0.1:8081 >>'$alog' 2>&1 && python3 '$H/bench_concurrency.py' --label '$label' --concurrency 2 --runs 3 --n-predict 256 --output '$CONC' --base http://127.0.0.1:8081 >>'$alog' 2>&1"
log "arm=$label split=1.10,0.90 ubatch=512 begin"
if env CUDA_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH="$PKG/bin:$PKG/lib:/usr/local/cuda-12.8/lib64" GGML_CUDA_TURING_CUBLAS_MIN_M=256 BENCH_CMD="$bench" HEALTH_TIMEOUT=600 "$H/run_variant.sh" "$label" -- "$PKG/bin/llama-server" "${base[@]}" -ub 512; then log "arm=$label PASS"; else log "arm=$label recoverable_FAIL"; fi
if grep -q "variant=$label NEW_XID_DETECTED" "$ROOT/logs/campaign.log"; then phase aborted-new-xid; exit 93; fi
phase disambig-done
