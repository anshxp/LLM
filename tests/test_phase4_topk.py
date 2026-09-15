import torch

from config.model_config import ModelConfig
from inference.generate import generate
from model.llm import LLM


def test_top_k_larger_than_vocabulary_is_supported():
    torch.manual_seed(4)
    config = ModelConfig(vocab_size=8, context_length=4, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0)
    model = LLM(config)
    prompt = torch.tensor([[1, 2]])

    output = generate(model, prompt, max_new_tokens=1, top_k=100)

    assert output.shape == (1, 3)
