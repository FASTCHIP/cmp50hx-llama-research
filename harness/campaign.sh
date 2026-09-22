#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/home/fastchip/cmp50hx-llama-research
LOG=$ROOT/logs/campaign.log
START_EPOCH=1790027578
MIN_END_EPOCH=1790056378
STAMP=20260921T215258Z
RAW=$ROOT/raw
msk(){ TZ=Europe/Moscow date --iso-8601=seconds; }
log(){ printf '%s %s\n' "$(msk)" "$*" >>"$LOG"; }
restore(){
  trap - EXIT INT TERM
  sudo -n systemctl stop llama-research.service 2>/dev/null || true
  sudo -n cp "$ROOT/rollback/llama-qwen.service.initial" /etc/systemd/system/llama-qwen.service
  sudo -n systemctl daemon-reload
  sudo -n systemctl restart llama-qwen.service
  "$ROOT/harness/wait_health.sh" http://127.0.0.1:8081/health 600 llama-qwen.service || true
  sudo -n systemctl stop bonsai-2.service 2>/dev/null || true
  log "campaign_restore health=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8081/health) llama_count=$(pgrep -x llama-server|wc -l)"
}
trap restore EXIT INT TERM
progress(){
 python3 - "$1" "$2" <<'PY'
import json,os,sys,time
p='/home/fastchip/cmp50hx-llama-research/summaries/progress.json';tmp=p+'.tmp';start=1790027578
x={'campaign_start_utc':'2026-09-21T21:52:58Z','minimum_end_utc':'2026-09-22T05:52:58Z','phase':sys.argv[1],'elapsed_seconds':int(time.time()-start),'last_result':sys.argv[2],'next_step':'continue plan matrix','hard_stop':None,'updated_epoch':int(time.time())}
open(tmp,'w').write(json.dumps(x,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,p)
PY
}
resume(){
 cat >"$ROOT/RESUME.md.tmp" <<EOF
# RESUME — CMP50HX llama.cpp research
- Campaign start UTC: 2026-09-21T21:52:58Z
- Minimum end UTC: 2026-09-22T05:52:58Z
- Current phase: $1
- Last successful variant: $2
- Active transaction: $3
- Hard-stop events: none
- Rollback: sudo -n cp $ROOT/rollback/llama-qwen.service.initial /etc/systemd/system/llama-qwen.service && sudo -n systemctl daemon-reload && sudo -n systemctl restart llama-qwen.service
EOF
 mv "$ROOT/RESUME.md.tmp" "$ROOT/RESUME.md"
}
heartbeat(){ while true; do sleep 1800; progress "$(cat "$ROOT/CURRENT_PHASE" 2>/dev/null||echo unknown)" "heartbeat; campaign running"; log 'heartbeat progress_json_updated'; done; }
heartbeat & HBPID=$!; trap 'kill $HBPID 2>/dev/null||true; restore' EXIT INT TERM
sudo -n systemctl stop bonsai-2.service 2>/dev/null || true
printf baseline >"$ROOT/CURRENT_PHASE"; progress baseline 'harness smoke passed'; resume baseline harness-smoke direct-prod
log 'phase=baseline begin'
BASE=$RAW/baseline-$STAMP.jsonl; CONC=$RAW/concurrency-$STAMP.jsonl
if [[ -f "$BASE" && -f "$CONC" && $(wc -l <"$BASE") -eq 30 && $(wc -l <"$CONC") -eq 9 ]]; then
  log 'baseline raw counts already complete; resuming after recoverable vision fixture failure'
else
  rm -f "$BASE" "$CONC"
  for c in short p4k p32k p64k; do runs=3; [[ $c == p64k ]] && runs=1; "$ROOT/harness/bench_request.py" --label prod-current --case "$c" --runs "$runs" --output "$BASE" --warmup; progress baseline "prod-current $c complete"; resume baseline "prod-current-$c" direct-prod; done
  for n in 1 2 5; do "$ROOT/harness/bench_concurrency.py" --label prod-current --concurrency "$n" --runs 3 --output "$CONC"; progress baseline "concurrency $n complete"; done
fi
"$ROOT/harness/vision_smoke.py" >"$ROOT/baseline/vision-$STAMP.json"
log 'baseline matrix complete; starting 20m soak'
TEL=$ROOT/hardware/telemetry-soak-$STAMP.csv
(while true;do printf '%s,' "$(date -u +%FT%TZ)";nvidia-smi --query-gpu=index,utilization.gpu,power.draw,clocks.current.graphics,clocks.current.memory,temperature.gpu --format=csv,noheader,nounits|tr '\n' ';';echo;sleep 1;done)>"$TEL" & TPID=$!
soak_end=$(( $(date +%s)+1200 )); SOAK=$RAW/soak-$STAMP.jsonl;rm -f "$SOAK";while (( $(date +%s)<soak_end ));do "$ROOT/harness/bench_request.py" --label prod-soak --case short --runs 1 --classes prose --output "$SOAK";done;kill $TPID||true;wait $TPID 2>/dev/null||true
progress baseline '20m soak complete';resume hardware prod-soak hardware-envelope
printf hardware >"$ROOT/CURRENT_PHASE";log 'phase=hardware stopping prod for microbench';sudo -n systemctl stop llama-qwen.service;until ! pgrep -x llama-server >/dev/null;do sleep 1;done
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64 "$ROOT/hardware/gpu_envelope" | tee "$ROOT/hardware/gpu-envelope.jsonl"
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64 "$ROOT/hardware/p2p_matrix" | tee "$ROOT/hardware/p2p-matrix.jsonl"
if grep -Eq '"mismatches":[1-9]' "$ROOT/hardware/p2p-matrix.jsonl";then log 'hard_stop P2P integrity mismatch';exit 93;fi
sudo -n systemctl restart llama-qwen.service;"$ROOT/harness/wait_health.sh" http://127.0.0.1:8081/health 600;progress hardware 'compute and P2P microbench complete';resume runtime hardware-envelope prod-current
BIN=/home/fastchip/llama.cpp-vbbb-b7d4d85-sm75/bin/llama-server
MODEL=/mnt/usbsata/models/Qwen3.8-27B-UD-Q4_K_XL.gguf
MMP=/mnt/usbsata/models/mmproj-Qwen3.8-27B-Q8_0.gguf
common=("$BIN" --model "$MODEL" --alias Qwen3.8-27B --mmproj "$MMP" --mmproj-offload --host 0.0.0.0 --port 8081 --parallel 1 --ctx-size 262144 --temp 0 --flash-attn on --gpu-layers 99 --tensor-split 1,1,1 --spec-type none -t 6 -tb 6 --api-key-file /etc/llama-server.api-key)
runv(){ label=$1;bench=$2;shift 2;printf "$label" >"$ROOT/CURRENT_PHASE";progress runtime "starting $label";resume runtime "$label pending" llama-research; if BENCH_CMD="$bench" "$ROOT/harness/run_variant.sh" "$label" -- "${common[@]}" "$@";then log "variant=$label PASS";resume runtime "$label PASS" prod-restored;else log "variant=$label recoverable_FAIL";fi;progress runtime "$label complete"; }
LONG=$RAW/longctx-$STAMP.jsonl;TOPO=$RAW/topology-$STAMP.jsonl;ELAS=$RAW/elastic-$STAMP.jsonl;rm -f "$LONG" "$TOPO" "$ELAS"
runv long-layer-q8 "$ROOT/harness/bench_request.py --label long-layer-q8 --case short --runs 3 --output $LONG --warmup; $ROOT/harness/bench_request.py --label long-layer-q8 --case p192k --runs 1 --output $LONG; $ROOT/harness/bench_request.py --label long-layer-q8 --case p250k --runs 1 --output $LONG; $ROOT/harness/vision_smoke.py > $ROOT/baseline/vision-long-layer-q8.json" --split-mode layer --cache-type-k q8_0 --cache-type-v q8_0 -b 2048 -ub 512
runv topo-tensor-q8 "$ROOT/harness/bench_request.py --label topo-tensor-q8 --case short --runs 2 --output $TOPO --warmup; $ROOT/harness/bench_request.py --label topo-tensor-q8 --case p32k --runs 1 --output $TOPO" --split-mode tensor --cache-type-k q8_0 --cache-type-v q8_0 -b 2048 -ub 512
runv topo-row-q8 "$ROOT/harness/bench_request.py --label topo-row-q8 --case short --runs 2 --output $TOPO --warmup; $ROOT/harness/bench_request.py --label topo-row-q8 --case p32k --runs 1 --output $TOPO" --split-mode row --cache-type-k q8_0 --cache-type-v q8_0 -b 2048 -ub 512
runv kv-q4q4 "$ROOT/harness/bench_request.py --label kv-q4q4 --case short --runs 2 --output $ELAS --warmup; $ROOT/harness/bench_request.py --label kv-q4q4 --case p250k --runs 1 --output $ELAS" --split-mode layer --cache-type-k q4_0 --cache-type-v q4_0 -b 2048 -ub 512
runv batch4096 "$ROOT/harness/bench_request.py --label batch4096 --case p32k --runs 1 --output $ELAS --warmup; $ROOT/harness/bench_request.py --label batch4096 --case short --runs 2 --output $ELAS" --split-mode layer --cache-type-k q8_0 --cache-type-v q8_0 -b 4096 -ub 512
runv ubatch1024 "$ROOT/harness/bench_request.py --label ubatch1024 --case p32k --runs 1 --output $ELAS --warmup; $ROOT/harness/bench_request.py --label ubatch1024 --case short --runs 2 --output $ELAS" --split-mode layer --cache-type-k q8_0 --cache-type-v q8_0 -b 2048 -ub 1024
# P2P-disabled diagnostic, same long-layer settings
NCCL_P2P_DISABLE=1 runv p2p-disabled "$ROOT/harness/bench_request.py --label p2p-disabled --case short --runs 2 --output $TOPO --warmup; $ROOT/harness/bench_request.py --label p2p-disabled --case p32k --runs 1 --output $TOPO" --split-mode layer --cache-type-k q8_0 --cache-type-v q8_0 -b 2048 -ub 512
# Keep collecting reproducibility data until the contractual minimum duration.
printf stability >"$ROOT/CURRENT_PHASE";log 'phase=stability repeat baseline until minimum_end'
while (( $(date +%s)<MIN_END_EPOCH ));do "$ROOT/harness/bench_request.py" --label stability-prod --case short --runs 1 --output "$RAW/stability-$STAMP.jsonl";progress stability 'one three-class stability block complete';resume stability stability-prod prod-current;done
log 'minimum eight hours reached';progress final 'minimum duration reached; final validation'
"$ROOT/harness/vision_smoke.py" >"$ROOT/baseline/vision-final.json"
"$ROOT/harness/bench_request.py" --label final-prod --case short --runs 1 --output "$RAW/final-$STAMP.jsonl"
log 'campaign_body_complete'
