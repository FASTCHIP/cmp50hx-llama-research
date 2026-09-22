#!/usr/bin/env python3
import argparse,concurrent.futures,hashlib,json,pathlib,time,urllib.request
from datetime import datetime,timezone
PROMPTS=['Объясни разницу между prefill и decode в LLM.','Напиши Python-код очереди с приоритетом.','Опиши безопасный rollback GPU-сервиса.','Сравни layer и tensor split.','Объясни проверку CUDA P2P.']
def one(base,key,prompt,n):
    data=json.dumps({'prompt':prompt,'n_predict':n,'temperature':0,'cache_prompt':False,'ignore_eos':True,'seed':1},ensure_ascii=False).encode(); req=urllib.request.Request(base+'/completion',data=data,headers={'Content-Type':'application/json','Authorization':'Bearer '+key}); t=time.monotonic()
    with urllib.request.urlopen(req,timeout=3600) as r: obj=json.loads(r.read())
    wall=time.monotonic()-t; tm=obj.get('timings') or {}; return {'wall_s':wall,'prompt_n':tm.get('prompt_n'),'predicted_n':tm.get('predicted_n'),'decode_tps':tm.get('predicted_per_second'),'prompt_tps':tm.get('prompt_per_second'),'finish_reason':obj.get('stop_type'),'content_sha256':hashlib.sha256(str(obj.get('content','')).encode()).hexdigest()}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--label',required=True); ap.add_argument('--concurrency',type=int,required=True); ap.add_argument('--runs',type=int,default=3); ap.add_argument('--output',required=True); ap.add_argument('--base',default='http://127.0.0.1:8081'); ap.add_argument('--api-key-file',default='/etc/llama-server.api-key'); ap.add_argument('--n-predict',type=int,default=256); a=ap.parse_args(); key=pathlib.Path(a.api_key_file).read_text().strip(); pathlib.Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    for rep in range(1,a.runs+1):
        start=time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as ex: rs=list(ex.map(lambda p:one(a.base,key,p,a.n_predict),PROMPTS[:a.concurrency]))
        elapsed=time.monotonic()-start; total=sum(r['predicted_n'] or 0 for r in rs); batch={'schema':1,'label':a.label,'case':f'conc{a.concurrency}','rep':rep,'ts_utc':datetime.now(timezone.utc).isoformat(),'batch_wall_s':elapsed,'aggregate_tps':total/elapsed,'requests':rs}
        with open(a.output,'a',encoding='utf-8') as f:f.write(json.dumps(batch,ensure_ascii=False,sort_keys=True)+'\n');f.flush()
        print(json.dumps(batch,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
