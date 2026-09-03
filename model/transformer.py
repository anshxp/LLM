import torch.nn as nn

from model.attention import CausalSelfAttention


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, dropout):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embedding_dim)

        self.attention = CausalSelfAttention(
            embedding_dim,
            num_heads,
            dropout,
        )

        self.layer_norm_2 = nn.LayerNorm(embedding_dim)

        self.feed_forward = FeedForward(
            embedding_dim,
            dropout,
        )

    def forward(self, x):
        x = x + self.attention(self.layer_norm_1(x))

        x = x + self.feed_forward(self.layer_norm_2(x))

        return x