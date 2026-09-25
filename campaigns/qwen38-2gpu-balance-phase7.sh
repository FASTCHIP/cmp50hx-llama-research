#!/usr/bin/env bash
# Phase 7: cap host prompt-cache RAM, then prove 256K + second client survive.
set -uo pipefail
ROOT=/home/fastchip/cmp50hx-llama-research
H=$ROOT/harness
OUT=${OUT:?campaign workspace}
LOG=$OUT/campaign.log
PHASE=$OUT/CURRENT_PHASE
UNIT=/etc/systemd/system/llama-qwen.service
NEW_RAM=${NEW_RAM:-4096}
msk(){ TZ=Europe/Moscow date --iso-8601=seconds; }
log(){ printf '%s %s\n' "$(msk)" "$*" | tee -a "$LOG"; }
phase(){ printf '%s\n' "$1" >"$PHASE"; log "phase=$1"; }

if ! grep -q -- "--cache-ram 16384" "$UNIT"; then
  log "phase7_note unit already at $(grep -o -- '--cache-ram [0-9-]*' "$UNIT")"
fi
sudo -n cp "$UNIT" "$OUT/state/llama-qwen.service.pre-ramcap"
sudo -n cp "$UNIT" "$ROOT/backups/llama-qwen.service.pre-ramcap-20260925"
sed -e "s|--cache-ram [0-9-]*|--cache-ram $NEW_RAM|" "$OUT/state/llama-qwen.service.pre-ramcap" >"$OUT/state/llama-qwen.service.ramcap"
grep -q -- "--cache-ram $NEW_RAM" "$OUT/state/llama-qwen.service.ramcap" || { log "phase7_abort sed_failed"; exit 70; }
sudo -n install -o root -g root -m 0644 "$OUT/state/llama-qwen.service.ramcap" "$UNIT"
sudo -n systemctl daemon-reload
sudo -n systemctl restart llama-qwen.service
if ! "$H/wait_health.sh" http://127.0.0.1:8081/health 900 llama-qwen.service; then
  log "phase7_restart_FAIL rollback"
  sudo -n cp "$OUT/state/llama-qwen.service.pre-ramcap" "$UNIT"
  sudo -n systemctl daemon-reload; sudo -n systemctl restart llama-qwen.service
  "$H/wait_health.sh" http://127.0.0.1:8081/health 900 llama-qwen.service && log "rollback_health_PASS"
  exit 71
fi
log "phase7 ramcap=$NEW_RAM health_PASS"
CURSOR_K=$(sudo -n journalctl -k --no-pager | wc -l)
printf '%s\n' "$CURSOR_K" >"$OUT/state/ramcap-kernel-cursor"
phase ramcap-live

# sample RSS of the live server while the sequence runs
( while :; do
    pid=$(pgrep -x llama-server | head -1)
    [ -n "${pid:-}" ] && { printf '%s %s %s\n' "$(date +%s)" "$(awk '/VmRSS/{print $2}' /proc/$pid/status 2>/dev/null)" "$(free -m | awk 'NR==2{print $7}')"; }
    sleep 10
  done ) >"$OUT/telemetry/ramcap-rss.txt" 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

python3 "$H/bench_request.py" --label ramcap-p250 --case p250k --runs 1 --classes prose \
  --output "$OUT/raw/ramcap.jsonl" --base http://127.0.0.1:8081 >>"$OUT/logs/ramcap.log" 2>&1
rc1=$?
log "ramcap p250k rc=$rc1"
python3 "$H/bench_concurrency.py" --label ramcap --concurrency 2 --runs 3 --n-predict 256 \
  --output "$OUT/raw/ramcap-conc.jsonl" --base http://127.0.0.1:8081 >>"$OUT/logs/ramcap.log" 2>&1
rc2=$?
log "ramcap conc2 rc=$rc2"
kill $SAMPLER 2>/dev/null || true
trap - EXIT

printf "health="; curl -sS --max-time 3 -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8081/health || true
printf "servers="; pgrep -x llama-server | wc -l
printf "peak RSS KB="; awk 'NR>1{if($2>m)m=$2}END{print m+0}' "$OUT/telemetry/ramcap-rss.txt"
printf "min available MB="; awk 'NR>2{if(m==0||$3<m)m=$3}END{print m+0}' "$OUT/telemetry/ramcap-rss.txt"
printf "oom events after cursor="; sudo -n journalctl -k --no-pager | tail -n "+$((CURSOR_K+1))" | grep -ci "Out of memory" || true
phase ramcap-tested
