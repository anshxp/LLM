from train import parse_args, validate_args


def test_valid_training_arguments_pass_validation():
    args = parse_args(["--batch-size", "1", "--gradient-accumulation-steps", "4", "--learning-rate", "0.0003", "--weight-decay", "0.01", "--epochs", "1", "--log-every", "10", "--max-grad-norm", "1"])
    validate_args(args)
