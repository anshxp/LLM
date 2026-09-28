from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 10000
    context_length: int = 256

    # RXLM 2: 256-dimensional representations with 6 attention heads.
    # The encoder + decoder + cross-attention architecture brings the
    # default model to approximately 12.5M trainable parameters.
    embedding_dim: int = 256
    num_layers: int = 4
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    num_heads: int = 6

    dropout: float = 0.1
