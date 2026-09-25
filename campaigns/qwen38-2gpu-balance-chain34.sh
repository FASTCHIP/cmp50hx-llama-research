#!/usr/bin/env bash
# Durable handoff: choose execution mode and run the p250k final gate.
set -uo pipefail
ROOT=/home/fastchip/cmp50hx-llama-research
while :; do
 state=$(systemctl show -p ActiveState --value qwen38-2gpu-chain23.service 2>/dev/null || true)
 case "$state" in active|activating|deactivating) sleep 30;; *) break;; esac
done
OUT=$(python3 - <<'PY'
from pathlib import Path
xs=list(Path('/home/fastchip/bench').glob('qwen38-2gpu-balance-*'))
print(max(xs,key=lambda p:p.stat().st_mtime) if xs else '')
PY
)
[[ -n "$OUT" && $(cat "$OUT/CHAIN_STATE" 2>/dev/null) == phase3-done ]] || exit 40
python3 - "$OUT" <<'PY'
import json,pathlib,statistics,sys
out=pathlib.Path(sys.argv[1])
rows=[json.loads(x) for x in (out/'raw/mode-finalists.jsonl').read_text().splitlines() if x.strip()]
conc=[json.loads(x) for x in (out/'raw/mode-finalists-conc.jsonl').read_text().splitlines() if x.strip()]
def med(label,case,key):
 v=[r[key] for r in rows if r.get('label')==label and r.get('case')==case and r.get('http_code')==200]
 return statistics.median(v) if v else None
def cmed(label):
 v=[r['aggregate_tps'] for r in conc if r.get('label')==label]
 return statistics.median(v) if v else None
lp,ld,lc=med('mode-layer','p120k','prompt_tps'),med('mode-layer','p120k','decode_tps'),cmed('mode-layer')
tp,td,tc=med('mode-tensor','p120k','prompt_tps'),med('mode-tensor','p120k','decode_tps'),cmed('mode-tensor')
mode='layer'
if None not in (lp,ld,lc,tp,td,tc) and tp>=.95*lp and tc>=.95*lc and td>=1.05*ld: mode='tensor'
split='1.00,1.00' if mode=='tensor' else (out/'state/selected-split.txt').read_text().strip()
ub=(out/'state/selected-ubatch.txt').read_text().strip()
doc={'mode':mode,'split':split,'ubatch':int(ub),'metrics':{'layer':{'p120_pp':lp,'p120_decode':ld,'conc2':lc},'tensor':{'p120_pp':tp,'p120_decode':td,'conc2':tc}}}
(out/'state/final-candidate.json').write_text(json.dumps(doc,indent=2)+'\n')
print(json.dumps(doc))
PY
MODE=$(python3 -c "import json; print(json.load(open('$OUT/state/final-candidate.json'))['mode'])")
SPLIT=$(python3 -c "import json; print(json.load(open('$OUT/state/final-candidate.json'))['split'])")
UBATCH=$(python3 -c "import json; print(json.load(open('$OUT/state/final-candidate.json'))['ubatch'])")
env OUT="$OUT" MODE="$MODE" SPLIT="$SPLIT" UBATCH="$UBATCH" "$ROOT/campaigns/qwen38-2gpu-balance-phase4.sh" || exit 41
printf '%s\n' phase4-done >"$OUT/CHAIN_STATE"
