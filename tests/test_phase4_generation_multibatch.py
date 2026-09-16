import torch

from config.model_config import ModelConfig
from inference.generate import generate
from model.llm import LLM


def test_generation_supports_multiple_prompts():
    config = ModelConfig(vocab_size=16, context_length=8, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0)
    model = LLM(config)
    prompts = torch.tensor([[1, 2], [3, 4]])
    output = generate(model, prompts, 2, top_k=2)
    assert output.shape == (2, 4)
