import pytest

from train import parse_args, validate_args


def test_non_positive_gradient_norm_is_rejected():
    args = parse_args(["--max-grad-norm", "0"])
    with pytest.raises(ValueError):
        validate_args(args)
