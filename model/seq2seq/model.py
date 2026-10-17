import torch
import torch.nn as nn
from model.seq2seq.config import Seq2SeqConfig
from model.seq2seq.transformer import EncoderBlock, DecoderBlock

class Seq2SeqLLM(nn.Module):
    """Encoder-decoder Transformer for corrupted-text reconstruction."""
    def __init__(self, config: Seq2SeqConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        self.encoder = nn.ModuleList([EncoderBlock(config.embedding_dim, config.num_heads, config.dropout) for _ in range(config.num_encoder_layers)])
        self.decoder = nn.ModuleList([DecoderBlock(config.embedding_dim, config.num_heads, config.dropout) for _ in range(config.num_decoder_layers)])
        self.encoder_norm = nn.LayerNorm(config.embedding_dim)
        self.decoder_norm = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def embed(self, ids):
        n = ids.shape[1]
        if n > self.config.context_length:
            raise ValueError(f"input length {n} exceeds context {self.config.context_length}")
        pos = torch.arange(n, device=ids.device)
        return self.token_embedding(ids) + self.position_embedding(pos)

    def encode(self, source_ids, source_mask=None):
        x = self.embed(source_ids)
        for block in self.encoder:
            x = block(x, source_mask)
        return self.encoder_norm(x)

    def decode(self, decoder_ids, memory, memory_mask=None):
        x = self.embed(decoder_ids)
        for block in self.decoder:
            x = block(x, memory, memory_mask)
        return self.lm_head(self.decoder_norm(x))

    def forward(self, source_ids, decoder_ids, source_mask=None):
        return self.decode(decoder_ids, self.encode(source_ids, source_mask), source_mask)

    @torch.no_grad()
    def generate(self, source_ids, max_new_tokens=128, bos_token_id=None, eos_token_id=None):
        self.eval()
        memory = self.encode(source_ids)
        start = self.config.bos_token_id if bos_token_id is None else bos_token_id
        output = torch.full((source_ids.size(0), 1), start, dtype=torch.long, device=source_ids.device)
        for _ in range(max_new_tokens):
            if output.shape[1] >= self.config.context_length:
                break
            next_token = self.decode(output, memory)[:, -1, :].argmax(dim=-1, keepdim=True)
            output = torch.cat([output, next_token], dim=1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
        return output
