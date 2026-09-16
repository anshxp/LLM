import pytest

from train import parse_args, validate_args


def test_negative_weight_decay_is_rejected():
    args = parse_args(["--weight-decay", "-0.1"])
    with pytest.raises(ValueError):
        validate_args(args)
