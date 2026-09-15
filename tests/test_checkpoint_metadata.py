import torch

from config.model_config import ModelConfig
from model.llm import LLM
from training.checkpoint import load_checkpoint


def test_old_style_checkpoint_still_loads(tmp_path):
    config = ModelConfig(vocab_size=16, context_length=8, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0)
    model = LLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "legacy.pt"
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "step": 3}, path)

    restored = LLM(config)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    assert load_checkpoint(restored, restored_optimizer, path) == 3
