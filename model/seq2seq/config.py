from dataclasses import dataclass
@dataclass
class Seq2SeqConfig:
    vocab_size: int = 10000
    context_length: int = 512
    embedding_dim: int = 128
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3
