"""Run a trained current-architecture classifier with its training-only vocabulary."""
import argparse,json,hashlib
from pathlib import Path
import torch
from train import build_model
from prepare_data import tokens
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--vocabulary',type=Path,required=True);p.add_argument('text');a=p.parse_args()
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True);vocab=json.loads(a.vocabulary.read_text());config=ck['config']
if ck.get('vocabulary_sha256') and hashlib.sha256(a.vocabulary.read_bytes()).hexdigest()!=ck['vocabulary_sha256']:raise ValueError('Vocabulary content mismatch')
if len(vocab)+2!=config['vocabulary']:raise ValueError('Vocabulary size mismatch')
ids=[vocab.get(w,1) for w in tokens(a.text)[:config['context']]] or [1]
model=build_model(config);model.load_state_dict(ck['model_state'],strict=True);model.eval()
with torch.no_grad():prob=model(torch.tensor([ids])).softmax(-1)[0].tolist()
print(json.dumps({'negative':prob[0],'positive':prob[1],'truncated':len(tokens(a.text))>config['context']}))
