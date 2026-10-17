import torch.nn as nn

from model.attention import CausalSelfAttention, CrossAttention, MultiHeadAttention


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


class EncoderBlock(nn.Module):
    """Bidirectional encoder block for source-to-target training."""
    def __init__(self, embedding_dim, num_heads, dropout):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, dropout)

    def forward(self, x):
        normalized = self.layer_norm_1(x)
        x = x + self.attention(normalized, normalized)
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.self_attention = CausalSelfAttention(embedding_dim, num_heads, dropout)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)
        self.cross_attention = CrossAttention(embedding_dim, num_heads, dropout)
        self.layer_norm_3 = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, dropout)

    def forward(self, x, encoder_output, causal_cross_attention=False):
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.cross_attention(
            self.layer_norm_2(x), encoder_output, causal=causal_cross_attention
        )
        x = x + self.feed_forward(self.layer_norm_3(x))
        return x


TransformerBlock = EncoderBlock
