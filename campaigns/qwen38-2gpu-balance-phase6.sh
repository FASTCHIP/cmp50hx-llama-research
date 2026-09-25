#!/usr/bin/env bash
# Phase 6: pick the final ubatch, promote the balanced profile, run client-path gates.
set -uo pipefail
ROOT=/home/fastchip/cmp50hx-llama-research
H=$ROOT/harness
OUT=${OUT:?campaign workspace}
LOG=$OUT/campaign.log
PHASE=$OUT/CURRENT_PHASE
PKG=/home/fastchip/llama.cpp-upstream-mmq-a02c7f5
UNIT=/etc/systemd/system/llama-qwen.service
SPLIT=1.10,0.90
msk(){ TZ=Europe/Moscow date --iso-8601=seconds; }
log(){ printf '%s %s\n' "$(msk)" "$*" | tee -a "$LOG"; }
while :; do
  state=$(systemctl show -p ActiveState --value qwen38-2gpu-p5.service 2>/dev/null || true)
  case "$state" in active|activating|deactivating) sleep 20;; *) break;; esac
done
[[ $(cat "$PHASE" 2>/dev/null) == disambig-done ]] || { log "phase6_abort marker=$(cat "$PHASE" 2>/dev/null)"; exit 60; }

UB=$(python3 - "$OUT" <<'PY'
import json,pathlib,statistics,sys
out=pathlib.Path(sys.argv[1])
def rows(name):
    p=out/'raw'/name
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
def p250(label):
    r=[x for x in rows('p250-final.jsonl')+rows('p250-disambig.jsonl') if x.get('label')==label and x.get('http_code')==200]
    return r[0] if r else None
def conc(label):
    return [x['aggregate_tps'] for x in rows('p250-disambig-conc.jsonl')+rows('mode-finalists-conc.jsonl') if x.get('label')==label]
def short_pp(label):
    v=[x['prompt_tps'] for x in rows('ubatch-screen.jsonl') if x.get('label')==label and x.get('case') in ('p4k','p32k') and x.get('http_code')==200]
    return statistics.median(v) if v else None
ref=p250('p250-ref'); u384=p250('p250-win'); u512=p250('p250-split512')
c384=conc('mode-layer'); c512=conc('p250-split512')
sp384=short_pp('ub384'); sp512=short_pp('ub512')
doc={'p250_1.00_ub512':ref,'p250_1.10_ub384':u384,'p250_1.10_ub512':u512,
     'conc2_ub384':c384,'conc2_ub512':c512,'shortmid_pp_ub384':sp384,'shortmid_pp_ub512':sp512,'decision':'512'}
ub='512'
if u384 and u512:
    long_gain=(u384['prompt_tps']-u512['prompt_tps'])/u512['prompt_tps']
    m384=statistics.median(c384) if c384 else None
    m512=statistics.median(c512) if c512 else None
    conc_ratio=(m384/m512) if (m384 and m512) else 1.0
    doc['long_gain_384_vs_512']=long_gain
    doc['conc_ratio_384_vs_512']=conc_ratio
    if long_gain>0.03 and conc_ratio>=0.95:
        ub='384'; doc['decision']='384'
(out/'state/final-choice.json').write_text(json.dumps(doc,indent=2,default=str)+'\n')
print(ub)
PY
)
log "phase6 final_ubatch=$UB split=$SPLIT"
cp "$UNIT" "$OUT/state/llama-qwen.service.pre-final" 2>/dev/null || sudo -n cp "$UNIT" "$OUT/state/llama-qwen.service.pre-final"
sudo -n cp "$UNIT" "/home/fastchip/cmp50hx-llama-research/backups/llama-qwen.service.pre-2gpu-balanced-20260925"
sed -e "s|--tensor-split [0-9.,]*|--tensor-split $SPLIT|" -e "s|-ub [0-9]*|-ub $UB|" "$OUT/state/llama-qwen.service.pre-final" >"$OUT/state/llama-qwen.service.new"
grep -q -- "--tensor-split $SPLIT" "$OUT/state/llama-qwen.service.new" || { log "phase6_abort split_not_applied"; exit 61; }
grep -q -- "-ub $UB" "$OUT/state/llama-qwen.service.new" || { log "phase6_abort ub_not_applied"; exit 61; }
sed -i 's|^Description=.*|Description=Qwen3.8 llama.cpp inference server (2x CMP50HX, balanced layer split, 262K Q8 KV)|' "$OUT/state/llama-qwen.service.new"
systemd-analyze verify "$OUT/state/llama-qwen.service.new" 2>&1 | grep -v "Unknown" | head -5 || true
sudo -n install -o root -g root -m 0644 "$OUT/state/llama-qwen.service.new" "$UNIT"
sudo -n systemctl daemon-reload
sudo -n systemctl restart llama-qwen.service
if "$H/wait_health.sh" http://127.0.0.1:8081/health 900 llama-qwen.service; then
  log "phase6 PROMOTE health_PASS"
else
  log "phase6 PROMOTE_FAIL rolling_back"
  sudo -n cp "$OUT/state/llama-qwen.service.pre-final" "$UNIT"
  sudo -n systemctl daemon-reload && sudo -n systemctl restart llama-qwen.service
  "$H/wait_health.sh" http://127.0.0.1:8081/health 900 llama-qwen.service && log "rollback_health_PASS"
  exit 62
fi
sleep 20

key=$(sudo -n cat /etc/llama-server.api-key)
python3 - "$OUT" "$key" <<'PY' | tee "$OUT/logs/acceptance.txt"
import json,pathlib,sys,urllib.request,urllib.error,subprocess
out=pathlib.Path(sys.argv[1]); key=sys.argv[2]
BASE="http://127.0.0.1:8081"
def call(path,payload,timeout=600,hdrs=None):
    req=urllib.request.Request(BASE+path,data=json.dumps(payload).encode(),
        headers={"Content-Type":"application/json","Authorization":"Bearer "+key,**(hdrs or {})})
    try:
        with urllib.request.urlopen(req,timeout=timeout) as r: return r.status,r.read()
    except urllib.error.HTTPError as e: return e.code,e.read()
res={}
s,b=call("/v1/chat/completions",{"model":"Qwen3.8-27B","messages":[{"role":"user","content":"Ответь одним словом: цвет чистого неба?"}],"temperature":0,"max_tokens":24})
res['text']=(s,json.loads(b)['choices'][0]['finish_reason'] if s==200 else b[:80].decode(errors='replace'))
# known-colour vision fixture: solid blue PNG generated deterministically
import zlib,struct,base64
def png(rgb,w=64,h=64):
    raw=b''.join(b'\x00'+bytes(rgb)*w for _ in range(h))
    def chunk(t,d):
        c=t+d; return struct.pack(">I",len(d))+c+struct.pack(">I",zlib.crc32(c)&0xffffffff)
    return (b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack(">IIBBBBB",w,h,8,2,0,0,0))
            +chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b''))
b64=base64.b64encode(png((0,0,255))).decode()
s,b=call("/v1/chat/completions",{"model":"Qwen3.8-27B","temperature":0,"max_tokens":24,
  "messages":[{"role":"user","content":[{"type":"text","text":"Каким одним словом называется этот цвет?"},
    {"type":"image_url","image_url":{"url":"data:image/png;base64,"+b64}}]}]})
res['vision']=(s,(json.loads(b)['choices'][0]['message'].get('content') or '').strip()[:40] if s==200 else b[:80].decode(errors='replace'))
body=b""
req=urllib.request.Request(BASE+"/v1/chat/completions",data=json.dumps({"model":"Qwen3.8-27B","stream":True,"max_tokens":16,"messages":[{"role":"user","content":"Скажи слово тест"}]}).encode(),
    headers={"Content-Type":"application/json","Authorization":"Bearer "+key})
chunks=0;done=False
with urllib.request.urlopen(req,timeout=600) as r:
    for line in r:
        if line.startswith(b"data: "):
            chunks+=1
            if b"[DONE]" in line: done=True
res['sse']=(200,{"chunks":chunks,"done":done})
req=urllib.request.Request(BASE+"/v1/chat/completions",data=json.dumps({"model":"Qwen3.8-27B","max_tokens":4,"messages":[{"role":"user","content":"hi"}]}).encode(),
    headers={"Content-Type":"application/json","Authorization":"Bearer wrong-key"})
try:
    with urllib.request.urlopen(req,timeout=60) as r: res['wrong_key']=(r.status,'unexpected 200')
except urllib.error.HTTPError as e: res['wrong_key']=(e.code,'')
pid=subprocess.run(["pgrep","-x","llama-server"],capture_output=True,text=True).stdout.split()
cmd=pathlib.Path("/proc/"+pid[0]+"/cmdline").read_bytes().replace(b'\0',b' ').decode() if pid else ""
res['servers']=len(pid)
res['split_in_cmd']="--tensor-split 1.10,0.90" in cmd
res['ub_in_cmd']="-ub 512" in cmd or "-ub 384" in cmd
print(json.dumps(res,ensure_ascii=False,indent=2))
(out/'state/acceptance.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n')
PY
log "phase6_complete"
