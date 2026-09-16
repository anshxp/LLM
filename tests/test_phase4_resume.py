from train import parse_args


def test_resume_path_is_parsed():
    args = parse_args(["--resume", "checkpoints/model_epoch_1.pt"])
    assert str(args.resume).endswith("model_epoch_1.pt")
