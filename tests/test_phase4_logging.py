import pytest

from train import parse_args, validate_args


def test_zero_log_interval_is_rejected():
    args = parse_args(["--log-every", "0"])
    with pytest.raises(ValueError):
        validate_args(args)
