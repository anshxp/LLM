import torch

from config.model_config import ModelConfig
from inference.generate import generate
from model.llm import LLM


def test_generation_supports_top_k_and_preserves_prompt():
    torch.manual_seed(42)
    config = ModelConfig(vocab_size=32, context_length=8, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0)
    model = LLM(config)
    prompt = torch.tensor([[1, 2, 3]])

    output = generate(model, prompt, max_new_tokens=5, temperature=1.0, top_k=4)

    assert output.shape == (1, 8)
    assert torch.equal(output[:, :3], prompt)
    assert output.dtype == torch.long


def test_generation_rejects_invalid_top_k():
    config = ModelConfig(vocab_size=16, context_length=8, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0)
    model = LLM(config)

    try:
        generate(model, torch.tensor([[1, 2]]), 1, top_k=0)
    except ValueError as exc:
        assert "top_k" in str(exc)
    else:
        raise AssertionError("Expected invalid top_k to be rejected")
