import pytest

from train import parse_args, validate_args


def test_zero_epochs_are_rejected():
    args = parse_args(["--epochs", "0"])
    with pytest.raises(ValueError):
        validate_args(args)
