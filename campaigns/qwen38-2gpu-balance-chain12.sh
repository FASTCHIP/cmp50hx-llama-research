#!/usr/bin/env bash
# Durable handoff: phase1 result -> long gate -> ubatch sweep.
set -uo pipefail
ROOT=/home/fastchip/cmp50hx-llama-research
H=$ROOT/harness
P1UNIT=qwen38-2gpu-balance-p1.service
while :; do
  state=$(systemctl show -p ActiveState --value "$P1UNIT" 2>/dev/null || true)
  case "$state" in active|activating|deactivating) sleep 20;; *) break;; esac
done
OUT=$(python3 - <<'PY'
from pathlib import Path
xs=list(Path('/home/fastchip/bench').glob('qwen38-2gpu-balance-*'))
print(max(xs,key=lambda p:p.stat().st_mtime) if xs else '')
PY
)
[[ -n "$OUT" && -f "$OUT/raw/split-screen.jsonl" ]] || exit 20
[[ $(cat "$OUT/CURRENT_PHASE" 2>/dev/null) == split-screen-done ]] || exit 21
python3 "$H/qwen38_2gpu_select.py" "$OUT/raw/split-screen.jsonl" --expected 3 --output "$OUT/state/split-selection.json" || exit 22
winner=$(python3 -c "import json; print(json.load(open('$OUT/state/split-selection.json'))['winner'])")
case "$winner" in
  split100) split=1.00,1.00;;
  split110) split=1.10,0.90;;
  split115) split=1.15,0.85;;
  split120) split=1.20,0.80;;
  *) exit 23;;
esac
printf '%s\n' "$split" >"$OUT/state/split-screen-winner.txt"
env OUT="$OUT" SPLIT_B="$split" "$ROOT/campaigns/qwen38-2gpu-balance-phase1b.sh" || exit 24
python3 - "$OUT" "$split" <<'PY'
import json,pathlib,sys
out=pathlib.Path(sys.argv[1]); challenger=sys.argv[2]
rows=[json.loads(x) for x in (out/'raw/split-long.jsonl').read_text().splitlines() if x.strip()]
d={r['label']:r for r in rows if r.get('http_code')==200}
b=d.get('split100-long'); c=d.get('splitbest-long')
chosen='1.00,1.00'
if b and c and c['prompt_tps'] >= .98*b['prompt_tps'] and c['decode_tps'] >= .98*b['decode_tps']:
    chosen=challenger
(out/'state/selected-split.txt').write_text(chosen+'\n')
print(chosen)
PY
selected=$(cat "$OUT/state/selected-split.txt")
env OUT="$OUT" SPLIT="$selected" "$ROOT/campaigns/qwen38-2gpu-balance-phase2.sh" || exit 25
printf '%s\n' phase2-done >"$OUT/CHAIN_STATE"
