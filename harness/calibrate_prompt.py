#!/usr/bin/env python3
import argparse,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from bench_request import calibrated_prompt
p=argparse.ArgumentParser();p.add_argument('--target',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--base',default='http://127.0.0.1:8081');p.add_argument('--api-key-file',default='/etc/llama-server.api-key');a=p.parse_args();key=pathlib.Path(a.api_key_file).read_text().strip();text,n=calibrated_prompt(a.base,a.target,'Provide a detailed technical conclusion.',key);pathlib.Path(a.output).write_text(text);print(n)
