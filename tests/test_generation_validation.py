import pytest
import torch

from config.model_config import ModelConfig
from inference.generate import generate
from model.llm import LLM


def test_generation_rejects_non_positive_temperature():
    model = LLM(ModelConfig(vocab_size=16, context_length=8, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0))
    with pytest.raises(ValueError):
        generate(model, torch.tensor([[1, 2]]), 1, temperature=0)
