import torch.nn as nn
from model.seq2seq.attention import SelfAttention,CrossAttention
class FeedForward(nn.Module):
    def __init__(self,dim,dropout):
        super().__init__(); self.net=nn.Sequential(nn.Linear(dim,4*dim),nn.GELU(),nn.Linear(4*dim,dim),nn.Dropout(dropout))
    def forward(self,x): return self.net(x)
class EncoderBlock(nn.Module):
    def __init__(self,dim,heads,dropout):
        super().__init__(); self.n1=nn.LayerNorm(dim); self.attn=SelfAttention(dim,heads,dropout,causal=False); self.n2=nn.LayerNorm(dim); self.ff=FeedForward(dim,dropout)
    def forward(self,x,mask=None):
        x=x+self.attn(self.n1(x),mask); return x+self.ff(self.n2(x))
class DecoderBlock(nn.Module):
    def __init__(self,dim,heads,dropout):
        super().__init__(); self.n1=nn.LayerNorm(dim); self.self_attn=SelfAttention(dim,heads,dropout,causal=True); self.n2=nn.LayerNorm(dim); self.cross_attn=CrossAttention(dim,heads,dropout); self.n3=nn.LayerNorm(dim); self.ff=FeedForward(dim,dropout)
    def forward(self,x,memory,memory_mask=None):
        x=x+self.self_attn(self.n1(x)); x=x+self.cross_attn(self.n2(x),memory,memory_mask); return x+self.ff(self.n3(x))
