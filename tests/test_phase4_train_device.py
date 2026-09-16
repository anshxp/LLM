from train import parse_args


def test_supported_device_options_parse():
    for value in ("auto", "cpu", "cuda"):
        assert parse_args(["--device", value]).device == value
