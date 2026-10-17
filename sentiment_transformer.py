"""Causal transformer classification over right-padded token batches."""
import torch
from torch import nn

class SentimentTransformer(nn.Module):
    def __init__(self,vocabulary=30522,width=128,heads=4,layers=4,context=256,pad_id=0):
        super().__init__()
        if min(vocabulary,width,heads,layers,context)<1 or width%heads:raise ValueError('Positive compatible model dimensions required')
        self.context,self.pad_id=context,pad_id
        self.tokens=nn.Embedding(vocabulary,width,padding_idx=pad_id)
        self.position=nn.Embedding(context,width)
        block=nn.TransformerEncoderLayer(width,heads,dim_feedforward=4*width,dropout=.1,batch_first=True)
        self.encoder=nn.TransformerEncoder(block,layers,enable_nested_tensor=False)
        self.classifier=nn.Linear(width,2)
    def forward(self,tokens):
        if tokens.ndim!=2 or not 0<tokens.shape[1]<=self.context:raise ValueError('Expected nonempty batch×sequence within context limit')
        padding=tokens.eq(self.pad_id)
        if padding.all(dim=1).any() or (padding[:,:-1]&~padding[:,1:]).any():raise ValueError('Each sequence needs valid tokens followed only by right padding')
        position=torch.arange(tokens.shape[1],device=tokens.device)
        features=self.tokens(tokens)+self.position(position)
        causal=torch.triu(torch.ones(tokens.shape[1],tokens.shape[1],device=tokens.device,dtype=torch.bool),diagonal=1)
        encoded=self.encoder(features,mask=causal,src_key_padding_mask=padding)
        final=(~padding).sum(dim=1)-1
        return self.classifier(encoded[torch.arange(len(tokens),device=tokens.device),final])
