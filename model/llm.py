import torch
import torch.nn as nn

from config.model_config import ModelConfig
from model.embeddings import TokenAndPositionEmbedding
from model.transformer import DecoderBlock, EncoderBlock


class LLM(nn.Module):
    """RXLM 2 encoder-decoder language model.

    The default training path can pass one token stream; that stream is used
    as both encoder input and decoder input, with a causal cross-attention
    mask preventing position t from seeing encoder positions greater than t.
    Separate encoder/decoder streams can also be supplied for seq2seq use.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embedding = TokenAndPositionEmbedding(
            config.vocab_size,
            config.context_length,
            config.embedding_dim,
        )

        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    config.embedding_dim,
                    config.num_heads,
                    config.dropout,
                )
                for _ in range(config.num_encoder_layers)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    config.embedding_dim,
                    config.num_heads,
                    config.dropout,
                )
                for _ in range(config.num_decoder_layers)
            ]
        )

        self.encoder_final_layer_norm = nn.LayerNorm(config.embedding_dim)
        self.decoder_final_layer_norm = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size)

    def forward(self, input_ids, decoder_input_ids=None):
        if decoder_input_ids is None:
            decoder_input_ids = input_ids
            causal_cross_attention = True
        else:
            causal_cross_attention = False

        encoder = self.embedding(input_ids)
        for block in self.encoder_blocks:
            encoder = block(encoder)
        encoder = self.encoder_final_layer_norm(encoder)

        decoder = self.embedding(decoder_input_ids)
        for block in self.decoder_blocks:
            decoder = block(
                decoder,
                encoder,
                causal_cross_attention=causal_cross_attention,
            )
        decoder = self.decoder_final_layer_norm(decoder)
        return self.lm_head(decoder)
