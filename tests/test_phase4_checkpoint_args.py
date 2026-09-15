from train import parse_args


def test_checkpoint_directory_can_be_changed():
    args = parse_args(["--checkpoint-dir", "artifacts"])
    assert str(args.checkpoint_dir).endswith("artifacts")
