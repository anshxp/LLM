import pytest

from train import parse_args, validate_args


def test_non_positive_max_batches_is_rejected():
    args = parse_args(["--max-train-batches", "0"])
    with pytest.raises(ValueError):
        validate_args(args)
