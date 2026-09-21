import torch

from config.model_config import ModelConfig
from model.llm import LLM


def test_tiny_model_can_overfit():
    torch.manual_seed(42)

    config = ModelConfig(
        vocab_size=16,
        context_length=8,
        embedding_dim=32,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
    )
    model = LLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]] * 4)
    target_ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]] * 4)

    model.train()
    initial_loss = None
    final_loss = None

    for step in range(300):
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = (
            torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
            )
        )
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert final_loss < initial_loss * 0.1
    assert final_loss < 0.2


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

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": 7,
            "epoch": 2,
            "batch_index": 0,
        },
        path,
    )

    restored = LLM(config)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    state = load_checkpoint(restored, restored_optimizer, path)

    assert state["step"] == 7
    assert state["epoch"] == 2
    assert state["best_validation_loss"] is None
    assert state["epochs_without_improvement"] == 0


def test_instruction_training_cli_accepts_instruction_dataset():
    from train import parse_args

    args = parse_args(["--dataset", "instruction", "--batch-size", "2"])

    assert args.dataset == "instruction"
    assert args.batch_size == 2

def test_instruction_training_cli_uses_sft_defaults():
    from train import DEFAULT_SFT_LEARNING_RATE, DEFAULT_SFT_LR_MIN, parse_args

    args = parse_args(["--dataset", "instruction"])

    assert args.learning_rate == DEFAULT_SFT_LEARNING_RATE
    assert args.lr_min == DEFAULT_SFT_LR_MIN
    assert str(args.pretrained_checkpoint).endswith("checkpoints\\phase7_run\\best_model.pt")

def test_instruction_training_cli_allows_sft_override():
    from train import parse_args

    args = parse_args([
        "--dataset", "instruction",
        "--learning-rate", "1e-5",
        "--lr-min", "1e-6",
    ])

    assert args.learning_rate == 1e-5
    assert args.lr_min == 1e-6
