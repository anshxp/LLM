import torch

from config.model_config import ModelConfig
from inference.generate import generate
from model.llm import LLM


def test_zero_new_tokens_returns_prompt_unchanged():
    config = ModelConfig(vocab_size=16, context_length=8, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0)
    model = LLM(config)
    prompt = torch.tensor([[1, 2, 3]])

    output = generate(model, prompt, max_new_tokens=0)

    assert torch.equal(output, prompt)
