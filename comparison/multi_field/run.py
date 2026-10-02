"""Run canonical GoEmotions, BANKING77, or CLINC150 structured requests per record."""
from __future__ import annotations
import argparse,hashlib,json,os,platform,subprocess,sys,tempfile,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from .adapters import DEFAULT_DSPY_MODEL,DSPyStructuredAdapter,JevStructuredAdapter,RLCDStructuredAdapter
from .djev_adapter import DjevStructuredAdapter
from .dataset import (fetch_banking77_sources, fetch_clinc150_sources, fetch_sources,
                      normalize, normalize_banking77, normalize_clinc150,
                      parse_banking77_csv, select_banking77_cases, select_cases)
from .metrics import percentile,summarize
from .models import StructuredPrediction

def atom(path:Path,value,lines=False):
 path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile('w',encoding='utf8',dir=path.parent,delete=False) as f:
  if lines:
   for x in value:f.write(json.dumps(x,sort_keys=True)+'\n')
  else:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
  tmp=f.name
 os.replace(tmp,path)
def parser():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--providers',required=True);p.add_argument('--dataset',choices=('goemotions','banking77','clinc150'),default='goemotions');p.add_argument('--limit',type=int,required=True);p.add_argument('--warmup',type=int,default=0);p.add_argument('--repetitions',type=int,default=1);p.add_argument('--out',type=Path,required=True);p.add_argument('--dspy-model',default=DEFAULT_DSPY_MODEL);p.add_argument('--jev-model',default='jev-1.13.0');p.add_argument('--djev-endpoint',default='',help='base URL of a self-hosted djev-run systemone service, e.g. http://twins:8080');p.add_argument('--allow-paid',action='store_true');p.add_argument('--dspy-no-thinking',action='store_true',help='send chat_template_kwargs.enable_thinking=false to the dspy arm (works on the pluto LiteLLM proxy for qwen thinking models)');return p
def git():
 try:return {'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
 except Exception:return {'commit':None}
def enum_token_lengths(choices):
 from core.engine_mlx import _enum_candidate_sequences,get_engine
 _model,tokenizer=get_engine();_prefix,candidates=_enum_candidate_sequences(tokenizer,'','  "intent": "',choices)
 lengths=Counter(len(continuation) for _,continuation in candidates)
 return {'histogram':dict(sorted(lengths.items())),'n_distinct_lengths':len(lengths),'predicted_sequential_forward_passes':1+len(lengths)}
def _djev_unavailable():
 # djev requires a live self-hosted endpoint (--djev-endpoint). Without one the
 # adapter must report 'unavailable' (no request is sent), mirroring how the
 # Jev arm reports missing_api_key and the DSPy arm reports dspy_not_installed.
 ad=DjevStructuredAdapter.__new__(DjevStructuredAdapter)
 ad.model='djev-dgemma';ad.endpoint='';ad.timeout_s=30.0;ad.steps=1
 class _U:
  provider='djev';model='djev-dgemma'
  def availability(self):return False
  def predict(self,case):return StructuredPrediction.unavailable('djev','djev-dgemma','no_djev_endpoint')
 return _U()
def main(argv=None,adapter_factories=None):
 a=parser().parse_args(argv);names=tuple(x.strip() for x in a.providers.split(',') if x.strip())
 if not names or set(names)-{'rlcd','dspy','jev','djev'} or len(set(names))!=len(names) or a.limit<=0 or a.warmup<0 or a.repetitions<=0:raise SystemExit('invalid providers or bounds')
 if a.dataset=='goemotions':
  test,labels,source=fetch_sources(a.out/'source');all_cases=normalize(test,labels);cases=select_cases(all_cases,a.limit);dataset={'name':'GoEmotions simplified test','source':source,'combined_sha256':hashlib.sha256(test+labels).hexdigest()};selection='first unseen positive example per canonical label in label order, then remaining test dataset order; deterministic';token_lengths=None
 elif a.dataset=='clinc150':
  test_parquet,card,source=fetch_clinc150_sources(a.out/'source');all_cases=normalize_clinc150(test_parquet,card);choices=all_cases[0].schema['intent']['choices'];cases=select_banking77_cases(all_cases,a.limit);dataset={'name':'CLINC150 OOS test','source':source,'combined_sha256':hashlib.sha256(test_parquet+card).hexdigest()};selection='random sample without replacement using fixed seed 20260919';token_lengths=enum_token_lengths(choices)
 else:
  script,train,test,source=fetch_banking77_sources(a.out/'source');all_cases=normalize_banking77(test,script);choices=all_cases[0].schema['intent']['choices'];train_categories={x[2] for x in parse_banking77_csv(train)};test_categories={x[2] for x in parse_banking77_csv(test)}
  if train_categories|test_categories != set(choices):raise ValueError(f'BANKING77 label mismatch: {sorted((train_categories|test_categories)^set(choices))}')
  cases=select_banking77_cases(all_cases,a.limit);dataset={'name':'BANKING77 test','source':source,'combined_sha256':hashlib.sha256(script+train+test).hexdigest()};selection='random sample without replacement using fixed seed 20260919';token_lengths=enum_token_lengths(choices)
 factories=adapter_factories or {'rlcd':lambda:RLCDStructuredAdapter(),'dspy':lambda:DSPyStructuredAdapter(a.dspy_model,no_thinking=a.dspy_no_thinking),'jev':lambda:JevStructuredAdapter(a.jev_model),'djev':(lambda:DjevStructuredAdapter(a.djev_endpoint) if a.djev_endpoint else _djev_unavailable())};configs={};rows=[];summaries={};started=time.perf_counter()
 for name in names:
  ad=factories[name]()
  if name=='jev' and not a.allow_paid:available,reason=False,'paid_provider_not_allowed'
  else:
   try:available=ad.availability()
   except Exception:available,reason=False,'availability_check_failed'
   reason=None if available else ('missing_api_key' if name=='jev' else 'adapter_unavailable')
  configs[name]={'model':ad.model,'available':available,'reason':reason}
  if not available:continue
  for c in cases[:a.warmup]:ad.predict(c)
  pairs=[];provider_started=time.perf_counter()
  for rep in range(a.repetitions):
   for c in cases:
    try:p=ad.predict(c)
    except Exception:p=StructuredPrediction.error(name,ad.model,'adapter_exception')
    pairs.append((c,p));rows.append({'case_id':c.case_id,'gold':c.gold,'repetition':rep,**p.to_dict()})
  summary=summarize(pairs);wall=time.perf_counter()-provider_started;latencies=[p.latency_ms for _,p in pairs if p.status=='ok' and p.latency_ms is not None]
  summary['client_latency_ms']={'n':len(latencies),'p50':percentile(latencies,50),'p90':percentile(latencies,90),'p95':percentile(latencies,95)};summary['observed_wall_clock']={'seconds':wall,'requests':len(pairs),'requests_per_second':len(pairs)/wall if wall else None};summaries[name]={ad.model:summary}
 support=Counter(case.gold[next(iter(case.gold))] for case in cases) if a.dataset in ('banking77','clinc150') else {field:sum(case.gold[field] for case in cases) for field in cases[0].schema}
 manifest={'dataset':dataset,'selected_case_ids':[c.case_id for c in cases],'selected_field_gold_support':({'count':len(support),'fields':sorted(support),'counts':dict(support)} if a.dataset in ('banking77','clinc150') else {'count':len([x for x in support if support[x]]),'fields':[x for x in support if support[x]],'counts':support}),'enum_token_lengths':token_lengths,'utc_started':datetime.now(timezone.utc).isoformat(),'providers':configs,'settings':{'providers':list(names),'dataset':a.dataset,'limit':a.limit,'warmup':a.warmup,'repetitions':a.repetitions,'concurrency':1,'allow_paid':a.allow_paid,'selection_strategy':selection},'platform':platform.platform(),'python':sys.version,'git':git(),'overall_wall_seconds':time.perf_counter()-started}
 atom(a.out/'manifest.json',manifest);atom(a.out/'results.jsonl',rows,True);atom(a.out/'summary.json',{'providers':summaries})
 for n,c in configs.items():print(f"{n}\t{c['model']}\t{'available' if c['available'] else 'unavailable'}")
 return 0
if __name__=='__main__':raise SystemExit(main())
