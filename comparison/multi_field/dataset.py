"""Canonical GoEmotions, BANKING77, and CLINC150 retrieval and normalization."""
from __future__ import annotations
import ast, csv, hashlib, io, json, re, random
from pathlib import Path
from urllib.request import Request, urlopen
from .models import StructuredCase
BASE='https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/'
TEST_URL=BASE+'test.tsv'; LABELS_URL=BASE+'emotions.txt'
BANKING77_SCRIPT_URL='https://huggingface.co/datasets/PolyAI/banking77/resolve/main/banking77.py'
BANKING77_TRAIN_URL='https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv'
BANKING77_TEST_URL='https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv'

def parse_labels(data: bytes) -> tuple[str, ...]:
 labels=tuple(line.strip() for line in data.decode('utf-8').splitlines() if line.strip())
 if len(labels)!=28 or len(set(labels))!=28: raise ValueError('expected exactly 28 unique canonical labels')
 return labels

def parse_test(data: bytes, labels: tuple[str, ...]) -> list[tuple[str,str,tuple[int,...]]]:
 rows=[]
 for line_no,line in enumerate(data.decode('utf-8').splitlines(),1):
  parts=line.split('\t')
  if len(parts)!=3 or not parts[0] or not parts[2]: raise ValueError(f'malformed test row {line_no}')
  try: ids=tuple(int(x) for x in parts[1].split(',') if x!='')
  except ValueError as exc: raise ValueError(f'malformed label ids at row {line_no}') from exc
  if not ids or len(set(ids)) != len(ids) or any(i<0 or i>=len(labels) for i in ids): raise ValueError(f'invalid label ids at row {line_no}')
  rows.append((parts[2],parts[0],ids))
 if not rows: raise ValueError('test data is empty')
 return rows

def normalize(test: bytes, label_bytes: bytes) -> list[StructuredCase]:
 labels=parse_labels(label_bytes); schema={x:{'type':'boolean','description':f'Whether the text expresses {x}.'} for x in labels}
 return [StructuredCase(case_id=f'goemotions-{cid}', context=text, schema=schema, gold={label:i in ids for i,label in enumerate(labels)}, metadata={'source_id':cid,'label_ids':list(ids)}) for cid,text,ids in parse_test(test,labels)]

def parse_banking77_labels(script: bytes) -> tuple[str,...]:
 tree=ast.parse(script.decode('utf-8'))
 for node in ast.walk(tree):
  if isinstance(node,ast.Call) and (getattr(node.func,'attr',None)=='ClassLabel' or getattr(node.func,'id',None)=='ClassLabel'):
   for keyword in node.keywords:
    if keyword.arg=='names':
     labels=tuple(ast.literal_eval(keyword.value))
     if len(labels)==77 and len(set(labels))==77 and all(isinstance(x,str) and x for x in labels): return labels
 raise ValueError('expected ClassLabel with exactly 77 unique canonical names')

def parse_banking77_csv(data: bytes) -> list[tuple[int,str,str]]:
 reader=csv.DictReader(io.StringIO(data.decode('utf-8'), newline=''))
 if reader.fieldnames != ['text','category']: raise ValueError(f'expected CSV header text,category; got {reader.fieldnames!r}')
 rows=[]
 for index,row in enumerate(reader):
  if set(row) != {'text','category'} or not row['text'] or not row['category']: raise ValueError(f'malformed BANKING77 row {index}')
  rows.append((index,row['text'],row['category']))
 if not rows: raise ValueError('BANKING77 CSV is empty')
 return rows

def normalize_banking77(test: bytes, script: bytes) -> list[StructuredCase]:
 labels=parse_banking77_labels(script); rows=parse_banking77_csv(test)
 categories={category for _,_,category in rows}
 if not categories <= set(labels): raise ValueError(f'BANKING77 test categories outside script labels: {sorted(categories-set(labels))}')
 schema={'intent':{'type':'enum','description':'BANKING77 intent category.', 'choices':list(labels)}}
 return [StructuredCase(case_id=f'banking77-test-{index}',context=text,schema=schema,gold={'intent':category},metadata={'source_id':index,'category':category}) for index,text,category in rows]

def _fetch(destination:Path, sources):
 destination.mkdir(parents=True,exist_ok=True); result=[]
 for name,url in sources:
  request=Request(url,headers={'User-Agent':'rlcd-engine-comparison'})
  with urlopen(request,timeout=30) as response: data=response.read();headers=dict(response.headers.items());final_url=response.url
  (destination/name).write_bytes(data);result.append({'name':name,'url':url,'final_url':final_url,'sha256':hashlib.sha256(data).hexdigest(),'response_headers':headers})
 return {x['name']:(destination/x['name']).read_bytes() for x in result},{'sources':result}

def fetch_sources(destination: Path) -> tuple[bytes,bytes,dict[str,object]]:
 data,source=_fetch(destination,(('test.tsv',TEST_URL),('emotions.txt',LABELS_URL)));return data['test.tsv'],data['emotions.txt'],source

def fetch_banking77_sources(destination:Path) -> tuple[bytes,bytes,bytes,dict[str,object]]:
 data,source=_fetch(destination,(('banking77.py',BANKING77_SCRIPT_URL),('train.csv',BANKING77_TRAIN_URL),('test.csv',BANKING77_TEST_URL)));return data['banking77.py'],data['train.csv'],data['test.csv'],source

def select_cases(cases: list[StructuredCase], limit: int | None) -> list[StructuredCase]:
 """First unseen positive per canonical field, then dataset order; deterministic."""
 if limit is None or limit>=len(cases): return cases
 selected=[]; seen=set()
 for field in cases[0].schema:
  for case in cases:
   if case.case_id not in seen and case.gold[field]: selected.append(case); seen.add(case.case_id); break
  if len(selected)==limit: return selected
 for case in cases:
  if case.case_id not in seen: selected.append(case); seen.add(case.case_id)
  if len(selected)==limit: break
 return selected

def select_banking77_cases(cases:list[StructuredCase],limit:int,seed:int=20260919)->list[StructuredCase]:
 if limit > len(cases): return cases
 return random.Random(seed).sample(cases,limit)

# ---------------------------------------------------------------------------
# CLINC150 / CLINC-OOS
# ---------------------------------------------------------------------------

CLINC_OOS_TEST_PARQUET_URL = 'https://huggingface.co/datasets/clinc/clinc_oos/resolve/main/imbalanced/test-00000-of-00001.parquet'
CLINC_OOS_HF_CARD_URL = 'https://huggingface.co/datasets/clinc/clinc_oos/raw/main/README.md'


def parse_clinc150_labels(card: bytes) -> tuple[str, ...]:
    """Extract the 151-way canonical intent label names from the HF card README.

    The card contains a YAML `dataset_info` block with `class_label.names`
    mapping integer index → intent name. We parse the first (imbalanced) block's
    mapping, which is the canonical 151-way set (150 in-domain intents + `oos`).
    Returns a tuple of 151 unique non-empty strings.
    """
    text = card.decode('utf-8')
    # Find the first class_label block after the first 'config_name:'
    first_block = text.split('config_name:', 1)[1] if 'config_name:' in text else text
    match = re.search(r"class_label:\s*\n\s*names:\s*\n((?:\s+'?\d+'?:\s+.+\n?)+)", first_block)
    if not match:
        raise ValueError('CLINC150: no class_label names block found in card')
    names_lines = match.group(1).splitlines()
    mapping: dict[int, str] = {}
    for line in names_lines:
        mm = re.match(r"\s+'?(\d+)'?:\s+(.+)$", line)
        if mm:
            mapping[int(mm.group(1))] = mm.group(2).strip().strip("'")
    if len(mapping) != 151:
        raise ValueError(f'CLINC150: card class_label block had {len(mapping)} entries, expected 151')
    labels = tuple(mapping[i] for i in range(151))
    if len(labels) != 151 or len(set(labels)) != 151 or any(not l for l in labels):
        raise ValueError(f'CLINC150: expected 151 unique non-empty labels, got {len(labels)} unique={len(set(labels))}')
    if 'oos' not in labels:
        raise ValueError('CLINC150: OOS label missing from canonical set')
    return labels


def _read_parquet_rows(data: bytes) -> list[tuple[str, str, int]]:
    """Read (index, text, intent_int) rows from a CLINC-OOS parquet file.

    Tries pyarrow in-process first; falls back to a subprocess call to the
    system python3 (which has pyarrow) when the venv lacks it. Returns
    [(source_id, text, intent_index), ...] in file order.
    """
    try:
        import pyarrow.parquet as pq
        df = pq.read_table(io.BytesIO(data)).to_pandas()
        return [(str(i), df.iloc[i]['text'], int(df.iloc[i]['intent'])) for i in range(len(df))]
    except ImportError:
        import subprocess, sys, tempfile, json as _json, os as _os
        with tempfile.NamedTemporaryFile('wb', suffix='.parquet', delete=False) as f:
            f.write(data); tmp_path = f.name
        script = (
            "import pyarrow.parquet as pq,sys,json;"
            "tbl=pq.read_table(sys.argv[1]);df=tbl.to_pandas();"
            "rows=[(str(i),df.iloc[i]['text'],int(df.iloc[i]['intent'])) for i in range(len(df))];"
            "print(json.dumps(rows))"
        )
        try:
            result = subprocess.run(['python3', '-c', script, tmp_path], capture_output=True, text=True, timeout=120)
        finally:
            _os.unlink(tmp_path)
        if result.returncode != 0:
            raise ValueError(f'CLINC150: parquet parse failed: {result.stderr[:500]}')
        return [tuple(row) for row in _json.loads(result.stdout)]


def parse_clinc150_parquet(data: bytes, labels: tuple[str, ...], reader=None) -> list[tuple[str, str, str]]:
    """Parse the CLINC-OOS test parquet file.

    The parquet has two columns: `text` (string) and `intent` (int64 index into
    the canonical 151-way class_label). Returns a list of
    (source_id, text, intent_name) tuples. `reader` is a pluggable
    bytes -> [(source_id, text, intent_int)] function (default:
    `_read_parquet_rows`), so tests can inject synthetic rows without pyarrow.
    """
    rows = (reader or _read_parquet_rows)(data)
    raw_rows = [(sid, text, int(idx)) for sid, text, idx in rows]
    index_set = set(idx for _, _, idx in raw_rows)
    if not index_set <= set(range(len(labels))):
        raise ValueError(f'CLINC150: parquet intent indices outside 0..{len(labels)-1}: {sorted(index_set - set(range(len(labels))))}')
    return [(sid, text, labels[idx]) for sid, text, idx in raw_rows]


def normalize_clinc150(test_parquet: bytes, card: bytes, reader=None) -> list[StructuredCase]:
    labels = parse_clinc150_labels(card)
    rows = parse_clinc150_parquet(test_parquet, labels, reader=reader)
    schema = {'intent': {'type': 'enum', 'description': 'CLINC150 intent class (150 in-domain intents + OOS).', 'choices': list(labels)}}
    return [StructuredCase(case_id=f'clinc150-{sid}', context=text, schema=schema, gold={'intent': intent_name},
                           metadata={'source_id': sid, 'intent_index': labels.index(intent_name)})
            for sid, text, intent_name in rows]


def fetch_clinc150_sources(destination: Path) -> tuple[bytes, bytes, dict[str, object]]:
    data, source = _fetch(destination, (('clinc_oos_test_imbalanced.parquet', CLINC_OOS_TEST_PARQUET_URL),
                                          ('clinc_oos_hf_card.md', CLINC_OOS_HF_CARD_URL)))
    return data['clinc_oos_test_imbalanced.parquet'], data['clinc_oos_hf_card.md'], source
