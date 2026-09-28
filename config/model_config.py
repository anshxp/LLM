from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 10000
    context_length: int = 256

    # RXLM 2: 256-dimensional representations with 4 attention heads.
    # The encoder + decoder + cross-attention architecture is retained.
    # 256 is divisible by 4, so each attention head receives 64 dimensions.
    embedding_dim: int = 256
    num_layers: int = 4
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    num_heads: int = 4

    dropout: float = 0.1
