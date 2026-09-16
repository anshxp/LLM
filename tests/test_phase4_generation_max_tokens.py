import inspect

from inference.generate import generate


def test_generation_requires_max_new_tokens_argument():
    assert inspect.signature(generate).parameters["max_new_tokens"].default is inspect.Parameter.empty
