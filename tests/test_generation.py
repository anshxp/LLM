import torch

from config.model_config import ModelConfig
from model.llm import LLM


def test_autoregressive_generation_shape_and_context():
    from inference.generate import generate

    config = ModelConfig(vocab_size=32, context_length=8, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0)
    model = LLM(config)
    model.eval()
    input_ids = torch.tensor([[1, 2, 3]])

    output = generate(model, input_ids, max_new_tokens=5)

    assert output.shape == (1, 8)
    assert output.dtype == torch.long
    assert torch.equal(output[:, :3], input_ids)
