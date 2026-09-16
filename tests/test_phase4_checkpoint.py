import torch

from config.model_config import ModelConfig
from model.llm import LLM
from training.checkpoint import save_checkpoint


def test_checkpoint_creates_parent_directory(tmp_path):
    config = ModelConfig(vocab_size=16, context_length=8, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0)
    model = LLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "nested" / "model.pt"

    save_checkpoint(model, optimizer, 1, path)

    assert path.exists()
