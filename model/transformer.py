import torch.nn as nn
from model.attention import SelfAttention,CausalSelfAttention,CrossAttention
class FeedForward(nn.Module):
    def __init__(self,embedding_dim,dropout):
        super().__init__(); self.network=nn.Sequential(nn.Linear(embedding_dim,4*embedding_dim),nn.GELU(),nn.Linear(4*embedding_dim,embedding_dim),nn.Dropout(dropout))
    def forward(self,x): return self.network(x)
class EncoderBlock(nn.Module):
    def __init__(self,embedding_dim,num_heads,dropout):
        super().__init__(); self.norm1=nn.LayerNorm(embedding_dim); self.attention=SelfAttention(embedding_dim,num_heads,dropout,causal=False); self.norm2=nn.LayerNorm(embedding_dim); self.ff=FeedForward(embedding_dim,dropout)
    def forward(self,x,key_padding_mask=None):
        x=x+self.attention(self.norm1(x),key_padding_mask); return x+self.ff(self.norm2(x))
class DecoderBlock(nn.Module):
    def __init__(self,embedding_dim,num_heads,dropout):
        super().__init__(); self.norm1=nn.LayerNorm(embedding_dim); self.self_attention=CausalSelfAttention(embedding_dim,num_heads,dropout); self.norm2=nn.LayerNorm(embedding_dim); self.cross_attention=CrossAttention(embedding_dim,num_heads,dropout); self.norm3=nn.LayerNorm(embedding_dim); self.ff=FeedForward(embedding_dim,dropout)
    def forward(self,x,encoder_output,encoder_padding_mask=None):
        x=x+self.self_attention(self.norm1(x)); x=x+self.cross_attention(self.norm2(x),encoder_output,encoder_padding_mask); return x+self.ff(self.norm3(x))
