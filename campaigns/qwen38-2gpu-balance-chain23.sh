#!/usr/bin/env bash
# Durable handoff: ubatch screen -> p120 gate -> layer/tensor comparison.
set -uo pipefail
ROOT=/home/fastchip/cmp50hx-llama-research
H=$ROOT/harness
while :; do
 state=$(systemctl show -p ActiveState --value qwen38-2gpu-chain12.service 2>/dev/null || true)
 case "$state" in active|activating|deactivating) sleep 30;; *) break;; esac
done
OUT=$(python3 - <<'PY'
from pathlib import Path
xs=list(Path('/home/fastchip/bench').glob('qwen38-2gpu-balance-*'))
print(max(xs,key=lambda p:p.stat().st_mtime) if xs else '')
PY
)
[[ -n "$OUT" && $(cat "$OUT/CHAIN_STATE" 2>/dev/null) == phase2-done ]] || exit 30
SPLIT=$(cat "$OUT/state/selected-split.txt")
python3 "$H/qwen38_2gpu_ubatch_select.py" "$OUT/raw/ubatch-screen.jsonl" --output "$OUT/state/ubatch-screen-selection.json" || exit 31
alt_label=$(python3 -c "import json; print(json.load(open('$OUT/state/ubatch-screen-selection.json'))['alternate'])")
case "$alt_label" in ub256) alt=256;; ub320) alt=320;; ub384) alt=384;; *) exit 32;; esac
printf '%s\n' "$alt" >"$OUT/state/ubatch-alternate.txt"
env OUT="$OUT" SPLIT="$SPLIT" ALT_UB="$alt" "$ROOT/campaigns/qwen38-2gpu-balance-phase2b.sh" || exit 33
python3 - "$OUT" "$alt" <<'PY'
import json,pathlib,sys
out=pathlib.Path(sys.argv[1]); alt=sys.argv[2]
rows=[json.loads(x) for x in (out/'raw/ubatch-long.jsonl').read_text().splitlines() if x.strip() and json.loads(x).get('http_code')==200]
g={}
for r in rows:g.setdefault(r['label'],[]).append(r)
def med(label,key):
 import statistics
 return statistics.median(x[key] for x in g[label])
def maxused(label):
 vals=[]
 for r in g[label]:
  d={int(p.split(',')[0]):int(p.split(',')[1]) for p in r['gpu_after'][:2]}
  vals.append(max(d.values()))
 return max(vals)
chosen='512'
if 'ub512-long' in g and 'ubalt-long' in g:
 bp,bd=med('ub512-long','prompt_tps'),med('ub512-long','decode_tps')
 ap,ad=med('ubalt-long','prompt_tps'),med('ubalt-long','decode_tps')
 if ad >= .95*bd and (ap > 1.02*bp or (ap >= .98*bp and maxused('ubalt-long')+256 < maxused('ub512-long'))): chosen=alt
(out/'state/selected-ubatch.txt').write_text(chosen+'\n')
print(chosen)
PY
UBATCH=$(cat "$OUT/state/selected-ubatch.txt")
env OUT="$OUT" SPLIT="$SPLIT" UBATCH="$UBATCH" "$ROOT/campaigns/qwen38-2gpu-balance-phase3.sh" || exit 34
printf '%s\n' phase3-done >"$OUT/CHAIN_STATE"
