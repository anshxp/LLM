import math

import torch
import torch.nn as nn

from config.seq2seq_model_config import Seq2SeqModelConfig


class TokenAndPositionEmbedding(nn.Module):
    """Shared token embedding with separate positional embeddings for each side."""

    def __init__(self, vocab_size, context_length, embedding_dim):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)

    def forward(self, input_ids):
        _, sequence_length = input_ids.shape
        if sequence_length > self.position_embedding.num_embeddings:
            raise ValueError(
                f"Sequence length ({sequence_length}) exceeds context length "
                f"({self.position_embedding.num_embeddings})"
            )

        positions = torch.arange(sequence_length, device=input_ids.device)
        return self.token_embedding(input_ids) + self.position_embedding(positions)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional causal masking and cross-attention."""

    def __init__(self, embedding_dim, num_heads, dropout, causal=False):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.causal = causal

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        batch_size, sequence_length, _ = x.shape
        return x.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def forward(self, query, key_value=None):
        if key_value is None:
            key_value = query

        batch_size, query_length, _ = query.shape
        _, key_length, _ = key_value.shape

        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key_value))
        v = self._split_heads(self.v_proj(key_value))

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        if self.causal:
            # The decoder may attend to itself and all earlier decoder tokens,
            # but never to future decoder tokens.
            mask = torch.triu(
                torch.ones(
                    query_length,
                    key_length,
                    device=query.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        output = weights @ v
        output = output.transpose(1, 2).contiguous().view(
            batch_size,
            query_length,
            self.embedding_dim,
        )
        return self.out_proj(output)


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, multiplier, dropout):
        super().__init__()
        hidden_dim = multiplier * embedding_dim
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)


class EncoderBlock(nn.Module):
    """Bidirectional Transformer encoder block."""

    def __init__(self, embedding_dim, num_heads, ff_multiplier, dropout):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.self_attention = MultiHeadAttention(
            embedding_dim, num_heads, dropout, causal=False
        )
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, ff_multiplier, dropout)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class DecoderBlock(nn.Module):
    """Causal decoder block with encoder-decoder cross-attention."""

    def __init__(self, embedding_dim, num_heads, ff_multiplier, dropout):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.self_attention = MultiHeadAttention(
            embedding_dim, num_heads, dropout, causal=True
        )
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)
        self.cross_attention = MultiHeadAttention(
            embedding_dim, num_heads, dropout, causal=False
        )
        self.layer_norm_3 = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, ff_multiplier, dropout)

    def forward(self, x, encoder_output):
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.cross_attention(
            self.layer_norm_2(x),
            encoder_output,
        )
        x = x + self.feed_forward(self.layer_norm_3(x))
        return x


class EncoderDecoderLLM(nn.Module):
    """Small encoder-decoder language model for the v2 architecture.

    The encoder has bidirectional self-attention. The decoder has causal
    self-attention followed by cross-attention over the encoder states.
    Encoder and decoder share the token embedding matrix; the output head is
    intentionally untied so the default configuration stays close to 11M
    parameters and remains easy to inspect during this project phase.
    """

    def __init__(self, config: Seq2SeqModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
        )
        self.encoder_position_embedding = nn.Embedding(
            config.context_length,
            config.embedding_dim,
        )
        self.decoder_position_embedding = nn.Embedding(
            config.context_length,
            config.embedding_dim,
        )

        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    config.embedding_dim,
                    config.num_heads,
                    config.feed_forward_multiplier,
                    config.dropout,
                )
                for _ in range(config.encoder_layers)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    config.embedding_dim,
                    config.num_heads,
                    config.feed_forward_multiplier,
                    config.dropout,
                )
                for _ in range(config.decoder_layers)
            ]
        )

        self.encoder_final_layer_norm = nn.LayerNorm(config.embedding_dim)
        self.decoder_final_layer_norm = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size)

    def _embed(self, input_ids, position_embedding):
        _, sequence_length = input_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"Sequence length ({sequence_length}) exceeds context length "
                f"({self.config.context_length})"
            )
        positions = torch.arange(sequence_length, device=input_ids.device)
        return self.token_embedding(input_ids) + position_embedding(positions)

    def encode(self, input_ids):
        x = self._embed(input_ids, self.encoder_position_embedding)
        for block in self.encoder_blocks:
            x = block(x)
        return self.encoder_final_layer_norm(x)

    def decode(self, decoder_input_ids, encoder_output):
        x = self._embed(decoder_input_ids, self.decoder_position_embedding)
        for block in self.decoder_blocks:
            x = block(x, encoder_output)
        x = self.decoder_final_layer_norm(x)
        return self.lm_head(x)

    def forward(self, input_ids, decoder_input_ids):
        encoder_output = self.encode(input_ids)
        return self.decode(decoder_input_ids, encoder_output)

    def num_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters())
