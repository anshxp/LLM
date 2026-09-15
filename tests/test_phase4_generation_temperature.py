import inspect

from inference.generate import generate


def test_temperature_default_is_one():
    assert inspect.signature(generate).parameters["temperature"].default == 1.0
