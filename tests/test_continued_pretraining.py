from pathlib import Path

import torch

from data.build_continued_corpus import _assert_allowed_path, _split_for_hash
from train_continued_pretraining import parse_args


def test_raw_folder_is_rejected():
    try:
        _assert_allowed_path(Path("data/raw/example.txt"))
    except ValueError:
        return
    raise AssertionError("data/raw must never be accepted as a continued-pretraining source")


def test_default_continued_pretraining_paths():
    args = parse_args([])
    assert args.pretrained_checkpoint == Path("checkpoints/phase7_run/best_model.pt")
    assert args.corpus_dir == Path("data/processed/continued_pretraining")
    assert args.checkpoint_dir == Path("checkpoints/continued_pretraining")
    assert not args.no_auto_resume


def test_split_assignment_is_deterministic():
    assert _split_for_hash("0" * 64) == _split_for_hash("0" * 64)
    assert _split_for_hash("f" * 64) == _split_for_hash("f" * 64)


def test_checkpoint_rng_state_is_supported(tmp_path):
    from config.model_config import ModelConfig
    from model.llm import LLM
    from training.checkpoint import load_checkpoint, save_checkpoint
    from training.optimizer import create_optimizer

    torch.manual_seed(123)
    model = LLM(ModelConfig())
    optimizer = create_optimizer(model, learning_rate=1e-4, weight_decay=0.0)
    path = tmp_path / "resume.pt"
    expected = torch.get_rng_state().clone()
    save_checkpoint(model, optimizer, 7, path, epoch=1, batch_index=4)
    torch.manual_seed(999)
    load_checkpoint(model, optimizer, path, map_location="cpu")
    assert torch.equal(torch.get_rng_state(), expected)
