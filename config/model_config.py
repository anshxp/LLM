from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 10000
    context_length: int = 256
    embedding_dim: int = 256
    num_layers: int = 4
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1

    def __post_init__(self):
        if self.context_length < 2:
            raise ValueError("context_length must be at least 2")
        if self.embedding_dim <= 0 or self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim must be positive and divisible by num_heads")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.num_encoder_layers <= 0 or self.num_decoder_layers <= 0:
            raise ValueError("Encoder and decoder must each have at least one layer")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
