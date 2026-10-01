import torch

from config.model_config import ModelConfig
from model.llm import LLM
from training.checkpoint import (
    load_checkpoint,
    save_checkpoint_with_history,
)


def _make_model_and_optimizer():
    config = ModelConfig(
        vocab_size=16,
        context_length=8,
        embedding_dim=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    model = LLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


def test_checkpoint_history_keeps_five_latest(tmp_path):
    model, optimizer = _make_model_and_optimizer()
    latest = tmp_path / "stage1" / "latest.pt"

    for step in range(1, 8):
        save_checkpoint_with_history(
            model,
            optimizer,
            step,
            latest,
            keep_last=5,
        )

    history = sorted(latest.parent.glob("checkpoint_step_*.pt"))
    assert [path.name for path in history] == [
        "checkpoint_step_000000003.pt",
        "checkpoint_step_000000004.pt",
        "checkpoint_step_000000005.pt",
        "checkpoint_step_000000006.pt",
        "checkpoint_step_000000007.pt",
    ]
    assert latest.exists()

    latest_data = torch.load(latest, map_location="cpu", weights_only=False)
    assert latest_data["step"] == 7


def test_corrupt_latest_falls_back_to_newest_valid_history(tmp_path):
    model, optimizer = _make_model_and_optimizer()
    latest = tmp_path / "stage1" / "latest.pt"

    save_checkpoint_with_history(model, optimizer, 10, latest, keep_last=5)
    save_checkpoint_with_history(model, optimizer, 11, latest, keep_last=5)

    latest.write_bytes(b"corrupt")

    restored_model, restored_optimizer = _make_model_and_optimizer()
    state = load_checkpoint(restored_model, restored_optimizer, latest)

    assert state == 11
    assert state["used_fallback"] is True
    assert state["checkpoint_path"].endswith(
        "checkpoint_step_000000011.pt"
    )
