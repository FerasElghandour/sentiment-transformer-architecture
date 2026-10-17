import torch
from sentiment_transformer import SentimentTransformer
torch.set_num_threads(2);model=SentimentTransformer(vocabulary=100,width=32,heads=4,layers=1,context=16).eval()
short=torch.tensor([[1,2,3],[4,5,6]])
padded=torch.cat([short,torch.zeros(2,3,dtype=torch.long)],dim=1)
a,b=model(short),model(padded)
assert a.shape==(2,2) and torch.allclose(a,b,atol=1e-6)
torch.nn.functional.cross_entropy(a,torch.tensor([0,1])).backward()
assert model.tokens.weight.grad.abs().sum()>0
try:model(torch.zeros(1,3,dtype=torch.long))
except ValueError:pass
else:raise AssertionError('All-padding review accepted')
try:model(torch.tensor([[1,0,2]]))
except ValueError:pass
else:raise AssertionError('Interior padding accepted')
print('Causal classifier shapes, right-padding invariance, finite gradients, and padding guards passed.')

try:SentimentTransformer(heads=0)
except ValueError:pass
else:raise AssertionError('Zero attention heads accepted')
print('Invalid attention dimensions are rejected explicitly.')
