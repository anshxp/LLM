from train import parse_args


def test_default_device_is_auto():
    assert parse_args([]).device == "auto"


def test_default_seed_is_deterministic_value():
    assert parse_args([]).seed == 42
