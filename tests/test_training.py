import pytest
import torch

from train import DEFAULT_PRETRAIN_CHECKPOINT


def test_instruction_training_cli_defaults():
    from train import parse_args

    args = parse_args(["--dataset", "instruction", "--batch-size", "2"])

    assert args.dataset == "instruction"
    assert args.batch_size == 2


def test_instruction_training_cli_uses_sft_defaults():
    from train import DEFAULT_SFT_LEARNING_RATE, DEFAULT_SFT_LR_MIN, parse_args

    args = parse_args(["--dataset", "instruction"])

    assert args.learning_rate == DEFAULT_SFT_LEARNING_RATE
    assert args.lr_min == DEFAULT_SFT_LR_MIN
    assert args.pretrained_checkpoint == DEFAULT_PRETRAIN_CHECKPOINT
    assert args.pretrained_checkpoint.as_posix().endswith(
        "checkpoints/phase7_run/best_model.pt"
    )


def test_instruction_training_cli_allows_sft_override():
    from train import parse_args

    args = parse_args([
        "--dataset", "instruction",
        "--learning-rate", "1e-5",
        "--lr-min", "1e-6",
    ])

    assert args.learning_rate == 1e-5
