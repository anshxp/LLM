import torch
import torch.nn as nn


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, context_length, embedding_dim):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = nn.Embedding(
            context_length,
            embedding_dim
        )

    def forward(self, input_ids):
        batch_size, sequence_length = input_ids.shape

        token_embeddings = self.token_embedding(input_ids)

        positions = torch.arange(
            sequence_length,
            device=input_ids.device
        )

        position_embeddings = self.position_embedding(positions)

        embeddings = token_embeddings + position_embeddings

        return embeddings