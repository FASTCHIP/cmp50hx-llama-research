#!/usr/bin/env python3
import json,pathlib,tempfile,unittest
from bench_request import extract_timings,append_row
from pivot import aggregate
class HarnessTests(unittest.TestCase):
 def test_extract_native(self):
  x=extract_timings({'timings':{'prompt_n':10,'prompt_ms':100,'predicted_n':20,'predicted_ms':200,'draft_n':8,'draft_n_accepted':4}});self.assertEqual(x['prompt_tps'],100);self.assertEqual(x['decode_tps'],100);self.assertEqual(x['draft_acceptance'],.5)
 def test_missing(self):
  x=extract_timings({});self.assertIsNone(x['decode_tps']);self.assertIsNone(x['prompt_n'])
 def test_aggregate(self):
  rows=[{'label':'a','case':'short','decode_tps':10,'prompt_tps':20,'predicted_n':100,'decode_ms':10000,'wall_s':11},{'label':'a','case':'short','decode_tps':20,'prompt_tps':40,'predicted_n':100,'decode_ms':5000,'wall_s':6}];x=aggregate(rows)[('a','short')];self.assertEqual(x['rows'],2);self.assertEqual(x['decode_median'],15);self.assertAlmostEqual(x['decode_aggregate'],200/15)
 def test_append(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'x.jsonl';append_row(p,{'x':1});append_row(p,{'x':2});self.assertEqual(len(p.read_text().splitlines()),2)
if __name__=='__main__':unittest.main()
