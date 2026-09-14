import torch
import torch.nn as nn


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, context_length, embedding_dim):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)

    def forward(self, input_ids):
        _, sequence_length = input_ids.shape
        if sequence_length > self.position_embedding.num_embeddings:
            raise ValueError(
                f"Sequence length ({sequence_length}) exceeds context length "
                f"({self.position_embedding.num_embeddings})"
            )

        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
        )

        return self.token_embedding(input_ids) + self.position_embedding(positions)
