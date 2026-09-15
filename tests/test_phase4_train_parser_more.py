from pathlib import Path

from train import parse_args


def test_training_parser_uses_path_for_checkpoint_directory():
    args = parse_args(["--checkpoint-dir", "checkpoints2"])
    assert isinstance(args.checkpoint_dir, Path)
