from __future__ import annotations
import json, tempfile, types, unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch
from comparison.multi_field.adapters import DSPyStructuredAdapter, RLCDStructuredAdapter
from comparison.multi_field.dataset import normalize, parse_labels, parse_test, select_cases
from comparison.multi_field.metrics import summarize
from comparison.multi_field.models import StructuredCase, StructuredPrediction
from comparison.multi_field.run import main
LABELS=b'admiration\namusement\nanger\nannoyance\napproval\ncaring\nconfusion\ncuriosity\ndesire\ndisappointment\ndisapproval\ndisgust\nembarrassment\nexcitement\nfear\ngratitude\ngrief\njoy\nlove\nnervousness\noptimism\npride\nrealization\nrelief\nremorse\nsadness\nsurprise\nneutral\n'
TEST=b'good\t0,27\tid1\nbad\t2\tid2\n'
class FakeDSPy:
 class InputField:
  def __init__(self,**kw):pass
 class OutputField(InputField):pass
 class JSONAdapter:pass
 def __init__(self):self.fields=None;self.calls=0;self.kw=None
 def Signature(self,f,i):self.fields=f;return f
 def configure_cache(self,**kw):pass
 def LM(self,**kw):self.kw=kw;return 'lm'
 def Predict(self,s):
  def call(**kw):self.calls+=1;return types.SimpleNamespace(**{k:False for k in s if k!='context'})
  return call
 def context(self,**kw):return nullcontext()
class MultiFieldTests(unittest.TestCase):
 def case(self):return normalize(TEST,LABELS)[0]
 def test_parse_mapping_and_selection(self):
  cases=normalize(TEST,LABELS);self.assertEqual(len(cases[0].schema),28);self.assertTrue(cases[0].gold['admiration']);self.assertTrue(cases[0].gold['neutral']);self.assertFalse(cases[0].gold['anger']);self.assertEqual([x.case_id for x in select_cases(cases,1)],['goemotions-id1'])
  with self.assertRaises(ValueError):parse_labels(b'x\n')
  with self.assertRaises(ValueError):parse_test(b'x\t99\tid\n',parse_labels(LABELS))
 def test_metrics_zero_division_and_mixed_cases(self):
  schema={'a':{'type':'boolean','description':'a'},'b':{'type':'boolean','description':'b'}}
  # all-true-negative a, false negative b, then false positive a with no a gold support in that row
  c1=StructuredCase('one','x',schema,{'a':False,'b':True}); c2=StructuredCase('two','x',schema,{'a':False,'b':False})
  p1=StructuredPrediction('x','m','ok',{'a':False,'b':False},latency_ms=10,exact_valid=True)
  p2=StructuredPrediction('x','m','ok',{'a':True,'b':False},latency_ms=30,exact_valid=True)
  s=summarize([(c1,p1),(c2,p2)])
  self.assertEqual((s['per_field']['a']['tp'],s['per_field']['a']['fp'],s['per_field']['a']['fn'],s['per_field']['a']['tn']),(0,1,0,1))
  self.assertEqual(s['per_field']['a']['f1'],0.0) # false-positive / zero gold support
  self.assertEqual(s['per_field']['b']['f1'],0.0) # false-negative / no predictions
  self.assertEqual(s['macro_f1_supported'],0.0); self.assertEqual(s['macro_f1_all_28'],0.0)
  invalid=StructuredPrediction.invalid('x','m','bad'); self.assertEqual(summarize([(c1,invalid)])['strict_schema_adherence'],0.0)
 def test_rlcd_one_full_schema_and_validation(self):
  captured={}
  def gen(context,schema,temperature):captured['schema']=schema;return {'parsed_json':{k:{'value':v} for k,v in self.case().gold.items()}}
  engine=types.ModuleType('core.engine');engine.run_rlcd_generation=gen; schema=types.ModuleType('core.schema');schema.StructuredSchema=lambda x:x
  with patch.dict('sys.modules',{'core.engine':engine,'core.schema':schema}):p=RLCDStructuredAdapter().predict(self.case())
  self.assertEqual(p.status,'ok');self.assertEqual(len(captured['schema']),28)
 def test_dspy_one_structured_call_temperature_zero_and_native_bools(self):
  fake=FakeDSPy();p=DSPyStructuredAdapter(dspy_module=fake).predict(self.case());self.assertEqual(p.status,'ok');self.assertEqual(fake.calls,1);self.assertEqual(len(fake.fields),29);self.assertEqual(fake.kw['temperature'],0)
 def test_runner_artifacts_support_and_latency(self):
  class Available:
   model='fake'
   def availability(self):return True
   def predict(self,c):return StructuredPrediction('rlcd','fake','ok',dict(c.gold),latency_ms=10,exact_valid=True)
  with tempfile.TemporaryDirectory() as t,patch('comparison.multi_field.run.fetch_sources',return_value=(TEST,LABELS,{'sources':[{'sha256':'x'}]})):
   out=Path(t)/'out';self.assertEqual(main(['--providers','rlcd','--limit','1','--out',str(out)],{'rlcd':lambda:Available()}),0)
   m=json.loads((out/'manifest.json').read_text());self.assertEqual(m['selected_field_gold_support']['fields'],['admiration','neutral']);self.assertEqual(m['selected_field_gold_support']['count'],2)
   s=json.loads((out/'summary.json').read_text())['providers']['rlcd']['fake'];self.assertEqual(s['client_latency_ms'],{'n':1,'p50':10,'p90':10,'p95':10});self.assertIn('requests_per_second',s['observed_wall_clock'])
  class Unavailable:
   model='none'
   def availability(self):return False
  with tempfile.TemporaryDirectory() as t,patch('comparison.multi_field.run.fetch_sources',return_value=(TEST,LABELS,{'sources':[{'sha256':'x'}]})):
   out=Path(t)/'out';main(['--providers','dspy','--limit','1','--out',str(out)],{'dspy':lambda:Unavailable()});self.assertEqual((out/'results.jsonl').read_text(),'')
if __name__=='__main__':unittest.main()
