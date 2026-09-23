#!/usr/bin/env bash
# Quality gate for the MMQ threshold: perplexity with the gate off vs on, same binary.
# Own transaction: snapshot -> stop serving -> two PPL runs -> restore exactly what was running.
set -uo pipefail
R=/home/fastchip/cmp50hx-llama-research
H=$R/harness
STATE=$R/state/quality-gate-initial.json
SNAP=$R/state/llama-qwen.service.quality-gate.snapshot
LOG=$R/logs/mmq-quality.log
PKG=/home/fastchip/mmq-builds/vbbb-mmq-thresh
LP="$PKG/bin:$PKG/lib:/usr/local/cuda-12.8/lib64"
M=/mnt/usbsata/models/Qwen3.8-27B-UD-Q4_K_XL.gguf
CORPUS=/home/fastchip/ppl-corpus.txt
COMMON=(-m "$M" -f "$CORPUS" -c 2048 -ngl 99 -ts 1,1,1 -fa on -ctk q8_0 -ctv q8_0 -t 6 --chunks 20 --seed 1234)

log(){ printf '%s %s\n' "$(TZ=Europe/Moscow date --iso-8601=seconds)" "$*" >>"$LOG"; }

restore(){
  local rc=$?
  trap - EXIT INT TERM
  log "restore_begin rc=$rc"
  if [[ -f $SNAP ]]; then sudo -n cp "$SNAP" /etc/systemd/system/llama-qwen.service; fi
  sudo -n systemctl daemon-reload
  sudo -n systemctl restart llama-qwen.service
  "$H/wait_health.sh" http://127.0.0.1:8081/health 480 llama-qwen.service || log "restore_health_FAIL"
  while read -r svc; do [[ -n $svc ]] && sudo -n systemctl start "$svc"; done < <(python3 "$H/rollback_policy.py" restore-list --state "$STATE" 2>/dev/null || true)
  sleep 5
  log "restore_done health=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 http://127.0.0.1:8081/health) llama_count=$(pgrep -x llama-server | wc -l) new_xid=$(python3 "$H/rollback_policy.py" xid-new --state "$STATE" 2>/dev/null || echo '?')"
  return $rc
}
trap restore EXIT INT TERM

python3 "$H/rollback_policy.py" snapshot --services llama-qwen.service,bonsai-2.service --out "$STATE" >>"$LOG" 2>&1
sudo -n cp /etc/systemd/system/llama-qwen.service "$SNAP"
log "quality_gate stage=stopping"
sudo -n systemctl stop bonsai-2.service 2>/dev/null || true
sudo -n systemctl stop llama-qwen.service
until ! ss -tln | grep -q ':8081 '; do sleep 1; done
sleep 5
echo "=== PPL: порог выключен ==="
env LD_LIBRARY_PATH="$LP" "$PKG/bin/llama-perplexity" "${COMMON[@]}" 2>&1 | grep -aE "Final estimate|calculating perplexity" | tail -2 | tee -a "$LOG"
echo "=== PPL: порог 256 ==="
env GGML_CUDA_TURING_CUBLAS_MIN_M=256 LD_LIBRARY_PATH="$LP" "$PKG/bin/llama-perplexity" "${COMMON[@]}" 2>&1 | grep -aE "Final estimate|calculating perplexity" | tail -2 | tee -a "$LOG"
echo "=== quality gate done ==="
