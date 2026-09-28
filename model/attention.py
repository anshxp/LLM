import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, attention_mask=None):
        batch_size, query_length, _ = query.shape
        key_length = key_value.shape[1]

        q = self.q_proj(query).view(
            batch_size, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(key_value).view(
            batch_size, key_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(key_value).view(
            batch_size, key_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        output = weights @ v
        output = output.transpose(1, 2).contiguous().view(
            batch_size, query_length, self.embedding_dim
        )
        return self.out_proj(output)


class CausalSelfAttention(MultiHeadAttention):
    def forward(self, x):
        sequence_length = x.shape[1]
        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool,
            )
        )[None, None, :, :]
        return super().forward(x, x, mask)


class CrossAttention(MultiHeadAttention):
    def forward(self, query, encoder_output, causal=True):
        query_length = query.shape[1]
        key_length = encoder_output.shape[1]
        if causal:
            # When the same stream is used for encoder and decoder pretraining,
            # prevent decoder position t from seeing encoder positions > t.
            mask = torch.tril(
                torch.ones(
                    query_length,
                    key_length,
                    device=query.device,
                    dtype=torch.bool,
                )
            )[None, None, :, :]
        else:
            mask = torch.ones(
                1, 1, query_length, key_length,
                device=query.device,
                dtype=torch.bool,
            )
        return super().forward(query, encoder_output, mask)
