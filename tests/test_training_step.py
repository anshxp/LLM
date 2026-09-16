import torch

from config.model_config import ModelConfig
from model.llm import LLM
from training.trainer import train_step


def test_train_step_returns_finite_loss():
    torch.manual_seed(1)
    config = ModelConfig(vocab_size=16, context_length=8, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0)
    model = LLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randint(0, config.vocab_size, (1, config.context_length))
    targets = torch.randint(0, config.vocab_size, (1, config.context_length))

    loss = train_step(model, optimizer, inputs, targets, max_grad_norm=1.0)

    assert torch.isfinite(torch.tensor(loss))
