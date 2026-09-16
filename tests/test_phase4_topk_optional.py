import torch

from config.model_config import ModelConfig
from inference.generate import generate
from model.llm import LLM


def test_generation_works_without_top_k():
    config = ModelConfig(vocab_size=8, context_length=4, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0)
    model = LLM(config)
    result = generate(model, torch.tensor([[1, 2]]), 1)
    assert result.shape == (1, 3)
