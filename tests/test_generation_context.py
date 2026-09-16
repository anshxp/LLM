import torch

from config.model_config import ModelConfig
from inference.generate import generate
from model.llm import LLM


def test_long_prompt_is_truncated_to_context_length():
    torch.manual_seed(3)
    config = ModelConfig(vocab_size=16, context_length=4, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0)
    model = LLM(config)
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6]])

    output = generate(model, prompt, max_new_tokens=2)

    assert output.shape == (1, 8)
    assert torch.equal(output[:, :6], prompt)
