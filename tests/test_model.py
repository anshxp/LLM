import torch

from config.model_config import ModelConfig
from model.llm import LLM


def test_forward_shape():
    config = ModelConfig(vocab_size=32, context_length=16, embedding_dim=32, num_layers=2, num_heads=4, dropout=0.0)
    model = LLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, config.context_length))

    logits = model(input_ids)

    assert logits.shape == (2, config.context_length, config.vocab_size)
    assert torch.isfinite(logits).all()


def test_causal_attention_blocks_future_tokens():
    config = ModelConfig(vocab_size=32, context_length=8, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0)
    model = LLM(config)
    model.eval()

    prefix = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    altered_future = prefix.clone()
    altered_future[0, 7] = 9

    with torch.no_grad():
        first = model(prefix)
        second = model(altered_future)

    assert torch.allclose(first[:, :7], second[:, :7], atol=1e-6, rtol=1e-5)
