import torch
import torch.nn as nn

from config.model_config import ModelConfig
from model.embeddings import TokenAndPositionEmbedding
from model.transformer import DecoderBlock, EncoderBlock


class Seq2SeqLanguageModel(nn.Module):
    """Small encoder-decoder LM trained to predict a future text segment.

    The encoder attends bidirectionally to a source prefix. The decoder uses
    causal self-attention and cross-attends to the encoder representation.
    Decoder targets are shifted by the dataset: decoder input is BOS + target[:-1].
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = TokenAndPositionEmbedding(
            config.vocab_size, config.context_length, config.embedding_dim
        )
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(config.embedding_dim, config.num_heads, config.dropout)
            for _ in range(config.num_encoder_layers)
        ])
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(config.embedding_dim, config.num_heads, config.dropout)
            for _ in range(config.num_decoder_layers)
        ])
        self.encoder_final_layer_norm = nn.LayerNorm(config.embedding_dim)
        self.decoder_final_layer_norm = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        # Share input/output token embeddings to reduce parameters; positional
        # embeddings are not tied to the output projection.
        self.lm_head.weight = self.embedding.token_embedding.weight

    def forward(self, input_ids, decoder_input_ids):
        if input_ids.ndim != 2 or decoder_input_ids.ndim != 2:
            raise ValueError("input_ids and decoder_input_ids must be [batch, sequence]")
        if input_ids.shape[0] != decoder_input_ids.shape[0]:
            raise ValueError("Encoder and decoder batch sizes must match")
        if input_ids.shape[1] > self.config.context_length:
            raise ValueError("Encoder sequence exceeds context_length")
        if decoder_input_ids.shape[1] > self.config.context_length:
            raise ValueError("Decoder sequence exceeds context_length")

        encoder = self.embedding(input_ids)
        for block in self.encoder_blocks:
            # EncoderBlock is intentionally bidirectional in this architecture.
            encoder = block(encoder)
        encoder = self.encoder_final_layer_norm(encoder)

        decoder = self.embedding(decoder_input_ids)
        for block in self.decoder_blocks:
            decoder = block(
                decoder,
                encoder,
                causal_cross_attention=False,
            )
        decoder = self.decoder_final_layer_norm(decoder)
        return self.lm_head(decoder)

    @torch.no_grad()
    def generate(self, source_ids, max_new_tokens=128, temperature=0.8, top_k=40,
                 eos_token_id=None):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        self.eval()
        encoder = self.embedding(source_ids[:, -self.config.context_length:])
        for block in self.encoder_blocks:
            encoder = block(encoder)
        encoder = self.encoder_final_layer_norm(encoder)

        generated = torch.full(
            (source_ids.shape[0], 1),
            fill_value=self.config_bos_id,
            dtype=torch.long,
            device=source_ids.device,
        )
        for _ in range(max_new_tokens):
            decoder = self.embedding(generated[:, -self.config.context_length:])
            for block in self.decoder_blocks:
                decoder = block(decoder, encoder, causal_cross_attention=False)
            logits = self.lm_head(self.decoder_final_layer_norm(decoder)[:, -1, :])
            logits = logits / temperature
            if top_k is not None and top_k > 0:
                k = min(int(top_k), logits.shape[-1])
                threshold = torch.topk(logits, k, dim=-1).values[:, -1, None]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
        return generated[:, 1:]

    @property
    def config_bos_id(self):
        # Special token ids are tokenizer-defined; set this before generation.
        if not hasattr(self, "_bos_token_id"):
            raise ValueError("Set model._bos_token_id from the active tokenizer before generation")
        return self._bos_token_id

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())
