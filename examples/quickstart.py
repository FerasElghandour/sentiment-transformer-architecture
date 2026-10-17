"""Offline example using the public implementation; see README for evidence scope."""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import torch
from sentiment_transformer import SentimentTransformer
torch.set_num_threads(2)
torch.manual_seed(0)
model=SentimentTransformer(vocabulary=100,width=32,heads=4,layers=1,context=16).eval()
short=torch.tensor([[1,2,3],[4,5,6]])
padded=torch.cat([short,torch.zeros(2,3,dtype=torch.long)],dim=1)
a,b=model(short),model(padded)
print({'logit_shape':list(a.shape),'padding_invariant':bool(torch.allclose(a,b,atol=1e-6))})
