import torch

from config.model_config import ModelConfig
from model.llm import LLM
from training.checkpoint import save_checkpoint


def test_checkpoint_records_requested_progress(tmp_path):
    config = ModelConfig(vocab_size=16, context_length=8, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0)
    model = LLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "progress.pt"
    save_checkpoint(model, optimizer, 5, path, epoch=3, batch_index=9)
    data = torch.load(path, weights_only=False)
    assert data["epoch"] == 3
    assert data["batch_index"] == 9
