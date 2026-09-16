import pytest

from train import parse_args, validate_args


def test_negative_workers_are_rejected():
    args = parse_args(["--num-workers", "-1"])
    with pytest.raises(ValueError):
        validate_args(args)
