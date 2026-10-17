import torch
import torch.nn as nn
from config.model_config import ModelConfig
from model.embeddings import TokenAndPositionEmbedding
from model.transformer import DecoderBlock,EncoderBlock
class LLM(nn.Module):
    """Encoder-decoder Transformer with span-denoising pretraining."""
    def __init__(self,config:ModelConfig):
        super().__init__(); self.config=config
        self.embedding=TokenAndPositionEmbedding(config.vocab_size,config.context_length,config.embedding_dim)
        self.encoder_blocks=nn.ModuleList([EncoderBlock(config.embedding_dim,config.num_heads,config.dropout) for _ in range(config.num_encoder_layers)])
        self.decoder_blocks=nn.ModuleList([DecoderBlock(config.embedding_dim,config.num_heads,config.dropout) for _ in range(config.num_decoder_layers)])
        self.encoder_final_layer_norm=nn.LayerNorm(config.embedding_dim); self.decoder_final_layer_norm=nn.LayerNorm(config.embedding_dim)
        self.lm_head=nn.Linear(config.embedding_dim,config.vocab_size,bias=False); self.lm_head.weight=self.embedding.token_embedding.weight
    def encode(self,input_ids,attention_mask=None):
        x=self.embedding(input_ids)
        for block in self.encoder_blocks: x=block(x,attention_mask)
        return self.encoder_final_layer_norm(x)
    def decode(self,decoder_input_ids,encoder_output,encoder_attention_mask=None):
        x=self.embedding(decoder_input_ids)
        for block in self.decoder_blocks: x=block(x,encoder_output,encoder_attention_mask)
        return self.lm_head(self.decoder_final_layer_norm(x))
    def forward(self,input_ids,decoder_input_ids,encoder_attention_mask=None):
        return self.decode(decoder_input_ids,self.encode(input_ids,encoder_attention_mask),encoder_attention_mask)
    @torch.no_grad()
    def generate(self,input_ids,max_new_tokens=128,bos_token_id=None,eos_token_id=None):
        self.eval(); memory=self.encode(input_ids); start=self.config.bos_token_id if bos_token_id is None else bos_token_id
        output=torch.full((input_ids.size(0),1),start,dtype=torch.long,device=input_ids.device)
        for _ in range(max_new_tokens):
            nxt=self.decode(output,memory)[:,-1,:].argmax(dim=-1,keepdim=True); output=torch.cat([output,nxt],dim=1)
            if eos_token_id is not None and torch.all(nxt==eos_token_id): break
        return output
