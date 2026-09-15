from train import DEFAULT_MAX_GRAD_NORM


def test_default_gradient_norm_is_positive():
    assert DEFAULT_MAX_GRAD_NORM > 0
