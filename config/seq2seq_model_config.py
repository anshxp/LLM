from dataclasses import dataclass


@dataclass
class Seq2SeqModelConfig:
    """Configuration for the v2 encoder-decoder Transformer.

    This configuration targets roughly 11M trainable parameters with the
    default values. The encoder uses bidirectional self-attention and the
    decoder uses causal self-attention plus encoder-decoder cross-attention.
    """

    vocab_size: int = 10_000
    context_length: int = 256

    embedding_dim: int = 256
    encoder_layers: int = 3
    decoder_layers: int = 3
    num_heads: int = 4
    feed_forward_multiplier: int = 4
    dropout: float = 0.1

    def __post_init__(self):
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if self.encoder_layers < 1 or self.decoder_layers < 1:
            raise ValueError("encoder_layers and decoder_layers must be >= 1")
        if self.feed_forward_multiplier < 1:
            raise ValueError("feed_forward_multiplier must be >= 1")
