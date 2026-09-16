from train import resolve_device


def test_cpu_selection_is_explicit():
    device = resolve_device("cpu")
    assert device.type == "cpu"
