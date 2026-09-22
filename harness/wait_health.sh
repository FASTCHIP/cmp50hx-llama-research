#!/usr/bin/env bash
set -Eeuo pipefail
url=${1:?health URL}; timeout=${2:-360}; unit=${3:-llama-qwen.service}
deadline=$((SECONDS+timeout))
while (( SECONDS < deadline )); do
  code=$(curl -sS -o /tmp/health.$$ -w '%{http_code}' --max-time 3 "$url" 2>/dev/null || true)
  if [[ "$code" == 200 ]]; then cat /tmp/health.$$; rm -f /tmp/health.$$; exit 0; fi
  sleep 2
done
rm -f /tmp/health.$$
echo "health timeout: $url after ${timeout}s" >&2
sudo -n systemctl status "$unit" --no-pager -l >&2 || true
sudo -n journalctl -u "$unit" --no-pager -n 120 >&2 || true
exit 1
