#!/usr/bin/env python3
import json,pathlib,tempfile,unittest
from bench_request import extract_timings,append_row
from pivot import aggregate
from rollback_policy import services_to_restore,journal_line_count,new_xid_count,parse_state
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

class RollbackPolicyTests(unittest.TestCase):
 """The runner must restore exactly the state it found, and judge Xids against a cursor."""
 def test_restores_only_services_active_before(self):
  both={'llama-qwen.service':True,'bonsai-2.service':True}
  self.assertEqual(services_to_restore(both,['llama-qwen.service','bonsai-2.service']),['llama-qwen.service','bonsai-2.service'])
  gpu3_off={'llama-qwen.service':True,'bonsai-2.service':False}
  self.assertEqual(services_to_restore(gpu3_off,['llama-qwen.service','bonsai-2.service']),['llama-qwen.service'])
 def test_unknown_service_is_not_started(self):
  self.assertEqual(services_to_restore({},['bonsai-2.service']),[])
  self.assertEqual(services_to_restore({'bonsai-2.service':False},['bonsai-2.service']),[])
 def test_pre_campaign_xids_are_not_new(self):
  old='a\nNVRM: Xid (PCI:0000:01:00): 43, pid=1, name=llama-server\nb\n'
  cursor=journal_line_count(old)
  self.assertEqual(new_xid_count(cursor,old),0)
  self.assertEqual(new_xid_count(cursor,old+'NVRM: Xid (PCI:0000:02:00): 43, pid=2, name=llama-server\n'),1)
  self.assertEqual(new_xid_count(cursor,old+'NVRM: Xid (PCI:0000:02:00): 43\nNVRM: Xid (PCI:0000:04:00): 43\n'),2)
 def test_cursor_counts_everything_when_zero(self):
  text='a\nNVRM: Xid (PCI:0000:01:00): 43\n'
  self.assertEqual(new_xid_count(0,text),1)
  self.assertEqual(new_xid_count(999,text),0)
 def test_cursor_survives_json_roundtrip(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'s.json'
   p.write_text(json.dumps({'services':{'llama-qwen.service':True},'journal_lines':7}))
   s=parse_state(p);self.assertEqual(s['journal_lines'],7);self.assertTrue(s['services']['llama-qwen.service'])
if __name__=='__main__':unittest.main()
