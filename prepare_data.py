"""Train-only vocabulary fitting for the canonical Stanford IMDB archive.

No tar extraction or pickle loading. Original train/test partitions are retained;
validation is a seeded stratified subset of original training. Budget limits are
explicit reduced-data experiments, not historical benchmark reproduction.
"""
import argparse,collections,hashlib,json,re,tarfile
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

def tokens(text):return re.findall(r"[a-z]+(?:'[a-z]+)?",text.lower())
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path);p.add_argument('--train-parquet',type=Path);p.add_argument('--test-parquet',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit-train',type=int);p.add_argument('--limit-test',type=int);p.add_argument('--vocabulary',type=int,default=20000);p.add_argument('--context',type=int,default=256);a=p.parse_args()
 if (a.limit_train is not None and a.limit_train<20) or (a.limit_test is not None and a.limit_test<2):raise ValueError('Subset budgets require at least20 training and2 test rows')
 if a.output.exists() or a.vocabulary<2 or a.context<1:raise ValueError('New output and positive token budget required')
 raw={s:[] for s in ('train','test')}
 if a.archive:
  if hashlib.md5(a.archive.read_bytes()).hexdigest()!='7c2ac02c03563afcf9b574c7e56c153a':raise ValueError('Unexpected official IMDB archive checksum')
  with tarfile.open(a.archive,'r:gz') as f:
   for member in sorted(f.getmembers(),key=lambda m:m.name):
    parts=member.name.split('/')
    if member.isfile() and len(parts)==4 and parts[1] in raw and parts[2] in ('pos','neg') and member.name.endswith('.txt'):
     if member.size>2_000_000:raise ValueError('Unexpected oversized review')
     raw[parts[1]].append((f.extractfile(member).read().decode('utf-8'),int(parts[2]=='pos')))
  source_hash={'archive':hashlib.sha256(a.archive.read_bytes()).hexdigest()}
 else:
  import pyarrow.parquet as pq
  expected={'train':'db47d16b5c297cc0dd625e519c81319c24c9149e70e8496de5475f6fa928342c','test':'b52e26e2f872d282ffac460bf9770b25ac6f102cda0e6ca7158df98c94e8b3da'}
  for split,path in [('train',a.train_parquet),('test',a.test_parquet)]:
   if path is None or hashlib.sha256(path.read_bytes()).hexdigest()!=expected[split]:raise ValueError('Supply the verified Stanford IMDB parquet files')
   table=pq.read_table(path).to_pydict();raw[split]=list(zip(table['text'],table['label']))
  source_hash=expected
 x,y=zip(*raw['train']);tr,va=train_test_split(np.arange(len(y)),test_size=.2,random_state=7,stratify=y)
 rng=np.random.default_rng(7);ti=rng.permutation(len(raw['test']))
 # Remove exact review overlap before vocabulary fitting; test rows are unchanged.
 test_texts={text for text,label in raw['test']}
 tr=[int(i) for i in tr if x[i] not in test_texts]
 train_texts={x[i] for i in tr}
 va=[int(i) for i in va if x[i] not in test_texts and x[i] not in train_texts]
 if a.limit_train:tr=tr[:a.limit_train];va=va[:max(20,a.limit_train//5)]
 if a.limit_test:ti=ti[:a.limit_test]
 counts=collections.Counter(t for i in tr for t in tokens(x[i]));vocab={w:i+2 for i,(w,n) in enumerate(sorted(counts.items(),key=lambda kv:(-kv[1],kv[0]))[:a.vocabulary-2])}
 def encode(text):
  ids=[vocab.get(w,1) for w in tokens(text)[:a.context]] or [1]
  return ids+[0]*(a.context-len(ids))
 data={}
 for split,indices,records in [('train',tr,raw['train']),('val',va,raw['train']),('test',ti,raw['test'])]:
  data['x_'+split]=np.array([encode(records[i][0]) for i in indices],dtype=np.int64);data['y_'+split]=np.array([records[i][1] for i in indices],dtype=np.int64)
 a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**data)
 a.output.with_suffix('.vocabulary.json').write_text(json.dumps(vocab,sort_keys=True)+'\n')
 a.output.with_suffix('.config.json').write_text(json.dumps({'classes':2,'vocabulary':len(vocab)+2,'width':32,'heads':4,'layers':2,'context':a.context},indent=2)+'\n')
 a.output.with_suffix('.provenance.json').write_text(json.dumps({'source':'Stanford Large Movie Review Dataset','source_sha256':source_hash,'seed':7,'training_indices':tr.tolist(),'validation_indices':va.tolist(),'test_indices':ti.tolist(),'vocabulary_fitted_only_on_training':True,'overlap_policy':'Exact text overlap excluded from train/validation using full official test texts; validation also excludes training texts; no test labels used','reduced_protocol':bool(a.limit_train or a.limit_test)},indent=2)+'\n')
if __name__=='__main__':main()
