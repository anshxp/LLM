import inspect

from training.trainer import train_step


def test_train_step_exposes_gradient_clipping():
    assert "max_grad_norm" in inspect.signature(train_step).parameters
