from train import parse_args, validate_args


def test_custom_optimizer_settings_are_accepted():
    args = parse_args(["--learning-rate", "0.0001", "--weight-decay", "0.02", "--log-every", "20"])
    validate_args(args)
    assert args.learning_rate == 0.0001
    assert args.weight_decay == 0.02
    assert args.log_every == 20
