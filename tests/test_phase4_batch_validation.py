import pytest

from train import parse_args, validate_args


def test_negative_batch_size_is_rejected():
    args = parse_args(["--batch-size", "-1"])
    with pytest.raises(ValueError):
        validate_args(args)
