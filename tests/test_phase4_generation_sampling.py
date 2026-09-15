import torch

from config.model_config import ModelConfig
from inference.generate import generate
from model.llm import LLM


def test_sampled_tokens_are_appended():
    torch.manual_seed(12)
    config = ModelConfig(vocab_size=16, context_length=8, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0)
    model = LLM(config)
    prompt = torch.tensor([[1, 2, 3]])
    output = generate(model, prompt, 2, temperature=0.8)
    assert output.shape[1] == prompt.shape[1] + 2
