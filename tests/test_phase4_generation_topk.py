import torch

from config.model_config import ModelConfig
from inference.generate import generate
from model.llm import LLM


def test_top_k_generation_stays_in_vocabulary():
    torch.manual_seed(9)
    config = ModelConfig(vocab_size=16, context_length=8, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0)
    model = LLM(config)
    output = generate(model, torch.tensor([[1, 2]]), 3, top_k=2)
    assert int(output.max()) < config.vocab_size
    assert int(output.min()) >= 0
