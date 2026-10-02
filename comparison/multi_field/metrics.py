"""Metrics for full Boolean records; scores are intentionally absent."""
from __future__ import annotations
import math
from collections import Counter
from .models import StructuredCase, StructuredPrediction

def percentile(values: list[float], q: float) -> float | None:
    if not values: return None
    values=sorted(values); pos=(len(values)-1)*q/100; lo=math.floor(pos); hi=math.ceil(pos)
    return values[lo] if lo==hi else values[lo]+(values[hi]-values[lo])*(pos-lo)

def summarize_enum(pairs:list[tuple[StructuredCase,StructuredPrediction]]) -> dict[str,object]:
    statuses=Counter(p.status for _,p in pairs); errors=Counter(p.error_code or p.status for _,p in pairs if p.status!='ok')
    evaluated=[(c,p) for c,p in pairs if p.status in ('ok','invalid')]; field=next(iter(pairs[0][0].schema)) if pairs else None
    choices=list(pairs[0][0].schema[field]['choices']) if field else []
    valid=sum(p.exact_valid for _,p in evaluated); correct=0; per={choice:Counter() for choice in choices}
    for case,pred in evaluated:
        gold=case.gold[field]; got=pred.prediction.get(field) if pred.status=='ok' and pred.exact_valid else None
        correct+=int(got==gold)
        for choice in choices:
            if got==choice and gold==choice: per[choice]['tp']+=1
            elif got==choice: per[choice]['fp']+=1
            elif gold==choice: per[choice]['fn']+=1
    def f1(c):
        precision=c['tp']/(c['tp']+c['fp']) if c['tp']+c['fp'] else 0.0
        recall=c['tp']/(c['tp']+c['fn']) if c['tp']+c['fn'] else 0.0
        return 2*precision*recall/(precision+recall) if precision+recall else 0.0
    per_out={choice:{**dict(c),'support':c['tp']+c['fn'],'f1':f1(c)} for choice,c in per.items()}
    supported=[x['f1'] for x in per_out.values() if x['support']>0]
    return {'n':len(pairs),'status_counts':dict(statuses),'attempted':sum(statuses[x] for x in ('ok','invalid','error')),'evaluated':len(evaluated),'strict_schema_adherence':valid/len(evaluated) if evaluated else None,'top1_accuracy':correct/len(evaluated) if evaluated else None,'macro_f1_present_gold_classes':sum(supported)/len(supported) if supported else None,'present_gold_class_count':len(supported),'per_class':per_out,'errors':dict(errors)}

def summarize(pairs:list[tuple[StructuredCase,StructuredPrediction]]) -> dict[str,object]:
    if pairs and all(spec['type']=='enum' for spec in pairs[0][0].schema.values()): return summarize_enum(pairs)
    statuses=Counter(p.status for _,p in pairs); errors=Counter(p.error_code or p.status for _,p in pairs if p.status!='ok')
    evaluated=[(c,p) for c,p in pairs if p.status in ('ok','invalid')]; fields=list(pairs[0][0].schema) if pairs else []
    valid=sum(p.exact_valid for _,p in evaluated); exact=0; tp=fp=fn=tn=0; per={f:Counter() for f in fields}
    for case,pred in evaluated:
        object_=pred.prediction if pred.status=='ok' and pred.exact_valid else {}
        exact += int(all(object_.get(f) == case.gold[f] for f in fields))
        for f in fields:
            gold=case.gold[f]; got=object_.get(f,False)
            if got and gold: tp+=1;per[f]['tp']+=1
            elif got: fp+=1;per[f]['fp']+=1
            elif gold: fn+=1;per[f]['fn']+=1
            else: tn+=1;per[f]['tn']+=1
    def prf(c):
        precision=c['tp']/(c['tp']+c['fp']) if c['tp']+c['fp'] else 0.0
        recall=c['tp']/(c['tp']+c['fn']) if c['tp']+c['fn'] else 0.0
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
        return {'tp':c['tp'],'fp':c['fp'],'fn':c['fn'],'tn':c['tn'],'precision':precision,'recall':recall,'f1':f1,'support':c['tp']+c['fn']}
    per_out={f:prf(per[f]) for f in fields}; micro=prf(Counter(tp=tp,fp=fp,fn=fn,tn=tn))
    supported=[x['f1'] for x in per_out.values() if x['support']>0]
    return {'n':len(pairs),'status_counts':dict(statuses),'attempted':sum(statuses[x] for x in ('ok','invalid','error')),'evaluated':len(evaluated),'strict_schema_adherence':valid/len(evaluated) if evaluated else None,'all_field_exact_record_match':exact/len(evaluated) if evaluated else None,'micro_f1':micro['f1'],'macro_f1_supported':sum(supported)/len(supported) if supported else None,'macro_f1_all_28':sum(x['f1'] for x in per_out.values())/len(fields) if fields else None,'per_field':per_out,'hamming_loss':(fp+fn)/(len(evaluated)*len(fields)) if evaluated and fields else None,'errors':dict(errors),'confusion_totals':{'tp':tp,'fp':fp,'fn':fn,'tn':tn}}
