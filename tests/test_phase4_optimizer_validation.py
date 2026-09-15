import pytest

from train import parse_args, validate_args


def test_zero_accumulation_is_rejected():
    args = parse_args(["--gradient-accumulation-steps", "0"])
    with pytest.raises(ValueError):
        validate_args(args)
