#!/usr/bin/env python3
import json,pathlib,struct,sys,urllib.request,zlib
# 32x32 opaque red PNG generated deterministically.
def chunk(kind,data):
    return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
w=h=32
raw=b''.join(b'\x00'+b'\xff\x00\x00'*w for _ in range(h))
png_bytes=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')
png=__import__('base64').b64encode(png_bytes).decode()
key=pathlib.Path('/etc/llama-server.api-key').read_text().strip();url=sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:8081/v1/chat/completions'
payload={'model':'Qwen3.8-27B','temperature':0,'max_tokens':64,'messages':[{'role':'user','content':[{'type':'text','text':'Какой основной цвет изображения? Ответь одним словом.'},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+png}}]}]}
r=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
with urllib.request.urlopen(r,timeout=600) as x:o=json.loads(x.read())
text=str(o['choices'][0]['message'].get('content','')).lower();print(json.dumps(o,ensure_ascii=False))
if not any(w in text for w in ('red','красн')):raise SystemExit('vision semantic check failed: '+text)
