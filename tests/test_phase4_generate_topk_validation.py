import inspect

from inference.generate import generate


def test_top_k_defaults_to_none():
    parameter = inspect.signature(generate).parameters["top_k"]
    assert parameter.default is None
