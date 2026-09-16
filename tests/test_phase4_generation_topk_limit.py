import torch

from config.model_config import ModelConfig
from inference.generate import generate
from model.llm import LLM


def test_top_k_one_generates_from_single_candidate_set():
    torch.manual_seed(5)
    config = ModelConfig(vocab_size=16, context_length=8, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0)
    model = LLM(config)
    prompt = torch.tensor([[1, 2]])
    output = generate(model, prompt, 1, top_k=1)
    assert output.shape == (1, 3)
