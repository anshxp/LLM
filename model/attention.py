import math
import torch
import torch.nn as nn
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout):
        super().__init__()
        if embedding_dim % num_heads: raise ValueError("embedding_dim must be divisible by num_heads")
        self.embedding_dim=embedding_dim; self.num_heads=num_heads; self.head_dim=embedding_dim//num_heads
        self.q_proj=nn.Linear(embedding_dim,embedding_dim); self.k_proj=nn.Linear(embedding_dim,embedding_dim)
        self.v_proj=nn.Linear(embedding_dim,embedding_dim); self.out_proj=nn.Linear(embedding_dim,embedding_dim); self.dropout=nn.Dropout(dropout)
    def forward(self,query,key_value,attention_mask=None):
        b,q_len,_=query.shape; k_len=key_value.shape[1]
        q=self.q_proj(query).view(b,q_len,self.num_heads,self.head_dim).transpose(1,2)
        k=self.k_proj(key_value).view(b,k_len,self.num_heads,self.head_dim).transpose(1,2)
        v=self.v_proj(key_value).view(b,k_len,self.num_heads,self.head_dim).transpose(1,2)
        scores=(q@k.transpose(-2,-1))/math.sqrt(self.head_dim)
        if attention_mask is not None: scores=scores.masked_fill(~attention_mask,torch.finfo(scores.dtype).min)
        out=self.dropout(torch.softmax(scores,dim=-1))@v
        return self.out_proj(out.transpose(1,2).contiguous().view(b,q_len,self.embedding_dim))
class SelfAttention(MultiHeadAttention):
    def __init__(self,embedding_dim,num_heads,dropout,causal=False):
        super().__init__(embedding_dim,num_heads,dropout); self.causal=causal
    def forward(self,x,key_padding_mask=None):
        n=x.shape[1]; mask=torch.ones(n,n,dtype=torch.bool,device=x.device)[None,None]
        if self.causal: mask=torch.tril(mask[0,0])[None,None]
        if key_padding_mask is not None: mask=mask & key_padding_mask[:,None,None,:].bool()
        return super().forward(x,x,mask)
class CausalSelfAttention(SelfAttention):
    def __init__(self,embedding_dim,num_heads,dropout): super().__init__(embedding_dim,num_heads,dropout,causal=True)
class CrossAttention(MultiHeadAttention):
    def forward(self,query,encoder_output,encoder_padding_mask=None):
        mask=None if encoder_padding_mask is None else encoder_padding_mask[:,None,None,:].bool()
        return super().forward(query,encoder_output,mask)
