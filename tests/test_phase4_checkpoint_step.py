import torch

from config.model_config import ModelConfig
from model.llm import LLM
from training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip_preserves_step(tmp_path):
    config = ModelConfig(vocab_size=16, context_length=8, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0)
    model = LLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "model.pt"
    save_checkpoint(model, optimizer, 21, path)
    restored = LLM(config)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    assert load_checkpoint(restored, restored_optimizer, path) == 21
