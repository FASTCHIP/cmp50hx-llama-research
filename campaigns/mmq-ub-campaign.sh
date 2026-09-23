#!/usr/bin/env bash
# MMQ/cuBLAS threshold A/B on the 5-slot production shape.
# Arms: gates -> control(unset) -> thr256 -> closing control -> cuBLAS reference.
# Every arm goes through harness/run_variant.sh, which restores the live unit afterwards.
set -uo pipefail
ROOT=/home/fastchip/cmp50hx-llama-research
H=$ROOT/harness
RAW=$ROOT/raw
STAMP=$(date -u +%Y%m%dT%H%M%SZ); LOG=/home/fastchip/cmp50hx-llama-research/logs/mmq-ub.log

PHASE=$ROOT/CURRENT_PHASE
JSONL=$RAW/mmq-ub-$STAMP.jsonl
CONC=$RAW/mmq-ub-conc-$STAMP.jsonl
GATES=$ROOT/logs/mmq-gates-$STAMP.log
PATCHPKG=/home/fastchip/mmq-builds/vbbb-mmq-thresh
CUBLASPKG=/home/fastchip/mmq-builds/vbbb-forcecublas
CACHE=/mnt/usbsata/models/kvcache-ab

msk(){ TZ=Europe/Moscow date --iso-8601=seconds; }
log(){ printf '%s %s\n' "$(msk)" "$*" >>"$LOG"; }
phase(){ printf '%s' "$1" >"$PHASE"; log "phase=$1"; }
progress(){ python3 - "$1" "$2" <<'PY'
import json,os,sys,time
p='/home/fastchip/cmp50hx-llama-research/summaries/progress-mmq.json'
x={'campaign':'mmq-ab','phase':sys.argv[1],'last':sys.argv[2],'updated':int(time.time())}
tmp=p+'.tmp';open(tmp,'w').write(json.dumps(x,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,p)
PY
}

MODEL=/mnt/usbsata/models/Qwen3.8-27B-UD-Q4_K_XL.gguf
MMP=/mnt/usbsata/models/mmproj-Qwen3.8-27B-Q8_0.gguf
DRAFT=/mnt/usbsata/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
common=(--model "$MODEL" --alias Qwen3.8-27B --mmproj "$MMP" --mmproj-offload
  --host 0.0.0.0 --port 8081 --parallel 5 --ctx-size 327680 --kv-unified --kv-unified-per-slot 65536
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0.0 --repeat-penalty 1.0
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 16384 --gpu-layers 99
  --split-mode layer --tensor-split 1,1,1 --reasoning-effort low --cache-disk "$CACHE"
  --cache-disk-max 102400 --spec-type none --spec-draft-n-max 7 --model-draft "$DRAFT"
  -t 6 -tb 6 -b 2048 -ub 512 --api-key-file /etc/llama-server.api-key)

arm(){ # arm LABEL PKGDIR ENVSTRING KIND
  local label=$1 pkg=$2 envs=$3 kind=$4
  phase "$label"; progress "$label" starting
  local lpath="$pkg/bin:$pkg/lib:/usr/local/cuda-12.8/lib64"
  local benchcmd="$H/mmq_bench.sh $label $JSONL $CONC $kind"
  [[ $kind == gates ]] && benchcmd="bash $GATES"
  local cmd
  if [[ -n $envs ]]; then cmd=(env "$envs" "$pkg/bin/llama-server" "${common[@]}")
  else cmd=("$pkg/bin/llama-server" "${common[@]}"); fi
  if env LD_LIBRARY_PATH="$lpath" BENCH_CMD="$benchcmd" HEALTH_TIMEOUT=900 "$H/run_variant.sh" "$label" -- "${cmd[@]}"; then
    log "arm=$label PASS"
  else
    log "arm=$label recoverable_FAIL"
  fi
  if grep -q "variant=$label NEW_XID_DETECTED" "$LOG"; then
    log "arm=$label ABORT_CAMPAIGN new_xid"; phase aborted-new-xid; exit 93
  fi
  progress "$label" done
}

log "=== mmq ub campaign start stamp=$STAMP ==="
mkdir -p /mnt/usbsata/models/kvcache-ab 2>/dev/null || true

cat >"$GATES" <<'GEOF'
set -uo pipefail
PKG=/home/fastchip/mmq-builds/vbbb-mmq-thresh
LP="$PKG/bin:$PKG/lib:/usr/local/cuda-12.8/lib64"
echo "== test-backend-ops MUL_MAT, threshold unset =="
env LD_LIBRARY_PATH="$LP" "$PKG/bin/test-backend-ops" test -b CUDA0 -o MUL_MAT >/tmp/ops-unset.log 2>&1
echo "rc_unset=$? lines=$(wc -l </tmp/ops-unset.log)"; tail -4 /tmp/ops-unset.log
echo "== test-backend-ops MUL_MAT, threshold forced (M>=1) =="
env GGML_CUDA_TURING_CUBLAS_MIN_M=1 LD_LIBRARY_PATH="$LP" "$PKG/bin/test-backend-ops" test -b CUDA0 -o MUL_MAT >/tmp/ops-forced.log 2>&1
echo "rc_forced=$? lines=$(wc -l </tmp/ops-forced.log)"; tail -4 /tmp/ops-forced.log
echo "== api smoke =="
curl -s -o /dev/null -w "health=%{http_code}\n" --max-time 10 http://127.0.0.1:8081/health
GEOF

arm mmq-ctl-ub2048 "$PATCHPKG" "" mid -ub 2048
arm mmq-thr256-ub2048 "$PATCHPKG" "GGML_CUDA_TURING_CUBLAS_MIN_M=256" mid -ub 2048
arm mmq-ctl2-ub2048 "$PATCHPKG" "" mid -ub 2048

phase done; progress done "campaign complete"
log "=== mmq ub campaign complete ==="
