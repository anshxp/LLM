from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 10000
    context_length: int = 256

    embedding_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4

    dropout: float = 0.1