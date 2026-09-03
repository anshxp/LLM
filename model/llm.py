import torch
import torch.nn as nn

from config.model_config import ModelConfig
from model.embeddings import TokenAndPositionEmbedding
from model.transformer import TransformerBlock


class LLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.embedding = TokenAndPositionEmbedding(
            config.vocab_size,
            config.context_length,
            config.embedding_dim,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.embedding_dim,
                    config.num_heads,
                    config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(
            config.embedding_dim
        )

        self.lm_head = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
        )

    def forward(self, input_ids):
        x = self.embedding(input_ids)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.final_layer_norm(x)

        logits = self.lm_head(x)

        return logits