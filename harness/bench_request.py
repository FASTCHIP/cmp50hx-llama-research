#!/usr/bin/env python3
"""Stdlib llama-server benchmark client; appends one JSON row/request."""
import argparse, hashlib, json, os, pathlib, statistics, subprocess, time, urllib.error, urllib.request
from datetime import datetime, timezone

PROMPTS = {
    "prose": "Напиши связное техническое объяснение, почему воспроизводимые измерения важны при оптимизации GPU-сервера. Не используй списки.",
    "code": "Напиши на Python функцию bounded_retry(fn, attempts, delay), использующую только стандартную библиотеку, с аннотациями типов и коротким примером.",
    "architecture": "Спроектируй безопасный цикл A/B-тестов llama.cpp: health gate, одна переменная, телеметрия, rollback trap и проверка целостности вывода.",
}
FILL = ("A production inference system must preserve request ordering, memory safety, reproducible telemetry, and bounded rollback. "
        "Engineers compare prompt processing, token generation, synchronization, memory bandwidth, and tail latency under controlled load. ")
TARGETS = {"short": None, "p4k": 4096, "p32k": 32768, "p64k": 63000, "p120k": 122880, "p192k": 196608, "p250k": 256000}

def http_json(url, payload, key, timeout=3600):
    data=json.dumps(payload, ensure_ascii=False).encode()
    req=urllib.request.Request(url, data=data, headers={"Content-Type":"application/json", "Authorization":f"Bearer {key}"})
    t=time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body=r.read(); code=r.status
    return code, json.loads(body), time.monotonic()-t

def tokenize(base, text, key):
    code,obj,_=http_json(base+"/tokenize", {"content":text}, key, 600)
    if code != 200: raise RuntimeError(f"tokenize HTTP {code}")
    toks=obj.get("tokens")
    if isinstance(toks,list): return len(toks)
    for k in ("count","n_tokens"):
        if isinstance(obj.get(k),int): return obj[k]
    raise RuntimeError(f"unknown tokenize response keys: {sorted(obj)}")

def calibrated_prompt(base, target, suffix, key):
    if target is None: return suffix, tokenize(base,suffix,key)
    unit=FILL
    unit_n=max(1, tokenize(base,unit,key))
    n=max(1, int((target-256)/unit_n))
    lo=max(1,n//2); hi=max(n+1,n*2)
    def make(k): return unit*k+"\n\n"+suffix
    while tokenize(base,make(hi),key) < target: lo,hi=hi,hi*2
    best=None
    for _ in range(24):
        mid=(lo+hi)//2; c=tokenize(base,make(mid),key)
        if best is None or abs(c-target)<abs(best[0]-target): best=(c,mid)
        if c<target: lo=mid+1
        else: hi=max(lo,mid-1)
    text=make(best[1]); count=tokenize(base,text,key)
    if abs(count-target) > target*0.01: raise RuntimeError(f"calibration outside 1%: {count} vs {target}")
    return text,count

def extract_timings(obj):
    t=obj.get("timings") or {}
    def val(*names):
        for n in names:
            if t.get(n) is not None: return t[n]
        return None
    pn=val("prompt_n","prompt_tokens"); pred=val("predicted_n","predicted_tokens")
    pps=val("prompt_per_second"); dps=val("predicted_per_second")
    pms=val("prompt_ms"); dms=val("predicted_ms")
    if pps is None and pn is not None and pms: pps=pn/(pms/1000)
    if dps is None and pred is not None and dms: dps=pred/(dms/1000)
    draft_n=val("draft_n","drafted_n") or 0; accepted=val("draft_n_accepted","draft_accepted_n") or 0
    return {"prompt_n":pn,"predicted_n":pred,"prompt_tps":pps,"decode_tps":dps,
            "prompt_ms":pms,"decode_ms":dms,"draft_n":draft_n,"draft_accepted":accepted,
            "draft_acceptance": (accepted/draft_n if draft_n else None)}

def gpu_snapshot():
    q="index,memory.used,pstate,power.draw,clocks.current.graphics,clocks.current.memory,utilization.gpu,temperature.gpu,pcie.link.gen.current,pcie.link.width.current"
    try:
        s=subprocess.check_output(["nvidia-smi",f"--query-gpu={q}","--format=csv,noheader,nounits"],text=True,timeout=10)
        return [x.strip() for x in s.splitlines()]
    except Exception as e: return [f"ERROR:{e}"]

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def append_row(path,row):
    pathlib.Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,"a",encoding="utf-8") as f:
        f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n"); f.flush(); os.fsync(f.fileno())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--label",required=True); ap.add_argument("--case",choices=TARGETS,required=True)
    ap.add_argument("--runs",type=int,default=3); ap.add_argument("--output",required=True); ap.add_argument("--base",default="http://127.0.0.1:8081")
    ap.add_argument("--api-key-file",default="/etc/llama-server.api-key"); ap.add_argument("--classes",default="prose,code,architecture")
    ap.add_argument("--warmup",action="store_true"); ap.add_argument("--binary",default="/proc/$(pgrep -x llama-server)/exe")
    a=ap.parse_args(); key=pathlib.Path(a.api_key_file).read_text().strip(); boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    prompts={}; target=TARGETS[a.case]
    for cls in a.classes.split(','):
        suffix=PROMPTS[cls]+" Ответ должен быть не короче 150 слов."
        prompts[cls]=calibrated_prompt(a.base,target,suffix,key)
    if a.warmup:
        http_json(a.base+"/completion",{"prompt":PROMPTS['prose'],"n_predict":16,"temperature":0,"cache_prompt":False,"ignore_eos":True},key,600)
    for cls,(prompt,actual) in prompts.items():
        for rep in range(1,a.runs+1):
            n_predict=512 if a.case in ("short","p4k","p32k") else 128
            payload={"prompt":prompt,"n_predict":n_predict,"temperature":0,"cache_prompt":False,"ignore_eos":True,"seed":1}
            before=gpu_snapshot(); code,obj,wall=http_json(a.base+"/completion",payload,key,7200); tm=extract_timings(obj)
            row={"schema":1,"label":a.label,"case":a.case,"prompt_class":cls,"rep":rep,"ts_utc":datetime.now(timezone.utc).isoformat(),
                 "boot_id":boot,"http_code":code,"wall_s":wall,"requested_prompt_tokens":target,"calibrated_prompt_tokens":actual,
                 "finish_reason":obj.get("stop_type") or obj.get("finish_reason"),"content_sha256":hashlib.sha256(str(obj.get('content','')).encode()).hexdigest(),
                 "gpu_before":before,"gpu_after":gpu_snapshot(),**tm}
            if tm["predicted_n"] is None or tm["decode_tps"] is None: raise RuntimeError(f"missing timings: {obj.keys()}")
            append_row(a.output,row); print(json.dumps(row,ensure_ascii=False),flush=True)
if __name__=="__main__": main()
