#!/usr/bin/env python3
import argparse,json,math,statistics

def percentile(v,p):
    s=sorted(v)
    if not s:return None
    i=(len(s)-1)*p; lo=math.floor(i); hi=math.ceil(i)
    return s[lo] if lo==hi else s[lo]+(s[hi]-s[lo])*(i-lo)
def aggregate(rows):
    groups={}
    for r in rows: groups.setdefault((r.get('label'),r.get('case')),[]).append(r)
    out={}
    for k,g in groups.items():
        d=[x['decode_tps'] for x in g if x.get('decode_tps') is not None]; p=[x['prompt_tps'] for x in g if x.get('prompt_tps') is not None]
        pred=sum(x.get('predicted_n') or 0 for x in g); sec=sum((x.get('decode_ms') or 0)/1000 for x in g)
        out[k]={'rows':len(g),'decode_median':statistics.median(d) if d else None,'decode_min':min(d) if d else None,'decode_p95':percentile(d,.95) if d else None,'decode_aggregate':pred/sec if sec else None,'prefill_median':statistics.median(p) if p else None,'wall_p95':percentile([x['wall_s'] for x in g],.95)}
    return out
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('file'); ap.add_argument('--expect',type=int); a=ap.parse_args()
    with open(a.file,encoding='utf-8') as f: rows=[json.loads(x) for x in f if x.strip()]
    print(f'row_count={len(rows)}')
    if a.expect is not None and len(rows)!=a.expect: raise SystemExit(f'row-count mismatch: expected {a.expect}, got {len(rows)}')
    for k,v in sorted(aggregate(rows).items()): print(json.dumps({'label':k[0],'case':k[1],**v},ensure_ascii=False,sort_keys=True))
if __name__=='__main__':main()
