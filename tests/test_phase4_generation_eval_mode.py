import torch

from config.model_config import ModelConfig
from inference.generate import generate
from model.llm import LLM


def test_generation_switches_model_to_eval_mode():
    config = ModelConfig(vocab_size=8, context_length=4, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.1)
    model = LLM(config)
    model.train()
    generate(model, torch.tensor([[1, 2]]), 1, top_k=1)
    assert not model.training
