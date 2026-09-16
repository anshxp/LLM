import torch

from config.model_config import ModelConfig
from inference.generate import generate
from model.llm import LLM


def test_generation_can_extend_beyond_context_length():
    config = ModelConfig(vocab_size=16, context_length=4, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0)
    model = LLM(config)
    prompt = torch.tensor([[1]])
    output = generate(model, prompt, 6, top_k=2)
    assert output.shape == (1, 7)
