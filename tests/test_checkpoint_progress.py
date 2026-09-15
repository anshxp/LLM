import torch

from config.model_config import ModelConfig
from model.llm import LLM
from training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_stores_epoch_and_batch_progress(tmp_path):
    config = ModelConfig(vocab_size=16, context_length=8, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0)
    model = LLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "progress.pt"

    save_checkpoint(model, optimizer, 7, path, epoch=2, batch_index=11)
    checkpoint = torch.load(path, weights_only=False)

    assert checkpoint["step"] == 7
    assert checkpoint["epoch"] == 2
    assert checkpoint["batch_index"] == 11

    restored = LLM(config)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    assert load_checkpoint(restored, restored_optimizer, path) == 7
