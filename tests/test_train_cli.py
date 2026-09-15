from train import parse_args, validate_args


def test_training_cli_accepts_smoke_run():
    args = parse_args(["--epochs", "1", "--max-train-batches", "2"])
    validate_args(args)
    assert args.epochs == 1
    assert args.max_train_batches == 2
