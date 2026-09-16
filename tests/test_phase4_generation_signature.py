import inspect

from inference.generate import generate


def test_generation_exposes_top_k_control():
    assert "top_k" in inspect.signature(generate).parameters
