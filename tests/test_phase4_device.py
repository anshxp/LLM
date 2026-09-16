import pytest

from train import resolve_device


def test_unknown_device_choice_is_not_silently_accepted():
    with pytest.raises((RuntimeError, ValueError)):
        resolve_device("invalid")
