import math
import torch
import torch.nn as nn
class MultiHeadAttention(nn.Module):
    def __init__(self,dim,heads,dropout):
        super().__init__()
        if dim%heads: raise ValueError("embedding_dim must divide evenly across heads")
        self.dim=dim; self.heads=heads; self.head_dim=dim//heads
        self.q=nn.Linear(dim,dim); self.k=nn.Linear(dim,dim); self.v=nn.Linear(dim,dim); self.o=nn.Linear(dim,dim); self.dropout=nn.Dropout(dropout)
    def forward(self,query,key_value,mask=None):
        b,ql,_=query.shape; kl=key_value.shape[1]
        q=self.q(query).view(b,ql,self.heads,self.head_dim).transpose(1,2)
        k=self.k(key_value).view(b,kl,self.heads,self.head_dim).transpose(1,2)
        v=self.v(key_value).view(b,kl,self.heads,self.head_dim).transpose(1,2)
        scores=(q@k.transpose(-2,-1))/math.sqrt(self.head_dim)
        if mask is not None:
            scores=scores.masked_fill(~mask,torch.finfo(scores.dtype).min)
        w=self.dropout(torch.softmax(scores,dim=-1))
        return self.o((w@v).transpose(1,2).contiguous().view(b,ql,self.dim))
class SelfAttention(MultiHeadAttention):
    def __init__(self,dim,heads,dropout,causal=False):
        super().__init__(dim,heads,dropout); self.causal=causal
    def forward(self,x,key_padding_mask=None):
        n=x.shape[1]; mask=torch.ones(n,n,dtype=torch.bool,device=x.device)[None,None]
        if self.causal: mask=torch.tril(mask[0,0])[None,None]
        if key_padding_mask is not None: mask=mask & key_padding_mask[:,None,None,:].bool()
        return super().forward(x,x,mask)
class CrossAttention(MultiHeadAttention):
    def forward(self,query,memory,memory_mask=None):
        mask=None if memory_mask is None else memory_mask[:,None,None,:].bool()
        return super().forward(query,memory,mask)
