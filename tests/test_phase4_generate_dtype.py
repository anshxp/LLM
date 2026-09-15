import torch

from config.model_config import ModelConfig
from inference.generate import generate
from model.llm import LLM


def test_generation_keeps_integer_token_dtype():
    config = ModelConfig(vocab_size=8, context_length=4, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0)
    model = LLM(config)
    result = generate(model, torch.tensor([[1, 2]], dtype=torch.long), 1, top_k=1)
    assert result.dtype == torch.long
