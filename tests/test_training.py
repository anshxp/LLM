import torch

from config.model_config import ModelConfig
from model.llm import LLM


def test_checkpoint_round_trip(tmp_path):
    from training.checkpoint import save_checkpoint, load_checkpoint

    torch.manual_seed(7)
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=4, eta_min=1e-4
    )
    path = tmp_path / "model.pt"

    input_ids = torch.randint(
        0, config.vocab_size, (2, config.context_length)
    )
    model.eval()
    with torch.no_grad():
        before = model(input_ids)

    optimizer.zero_grad(set_to_none=True)
    optimizer.step()
    scheduler.step()
    save_checkpoint(
        model,
        optimizer,
        12,
        path,
        epoch=3,
        scheduler=scheduler,
        best_validation_loss=2.5,
        epochs_without_improvement=1,
    )

    restored = LLM(config)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        restored_optimizer, T_max=4, eta_min=1e-4
    )
    state = load_checkpoint(
        restored,
        restored_optimizer,
        path,
        scheduler=restored_scheduler,
    )

    restored.eval()
    with torch.no_grad():
        after = restored(input_ids)

    assert state["step"] == 12
    assert state["epoch"] == 3
    assert state["best_validation_loss"] == 2.5
    assert state["epochs_without_improvement"] == 1
    assert torch.allclose(before, after, atol=1e-6, rtol=1e-5)
    assert restored_scheduler.state_dict() == scheduler.state_dict()


def test_legacy_checkpoint_remains_loadable(tmp_path):
    from training.checkpoint import save_checkpoint, load_checkpoint

    torch.manual_seed(11)
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
    path = tmp_path / "legacy.pt"

    torch.save({"model_state_dict": model.state_dict(), "step": 5}, path)

    restored = LLM(config)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    state = load_checkpoint(restored, restored_optimizer, path)

    assert state["step"] == 5
