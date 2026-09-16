import pytest

from train import parse_args, validate_args


def test_negative_learning_rate_is_rejected():
    args = parse_args(["--learning-rate", "-1"])
    with pytest.raises(ValueError):
        validate_args(args)
