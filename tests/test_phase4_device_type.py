from train import resolve_device


def test_resolved_device_is_torch_device():
    device = resolve_device("cpu")
    assert device.__class__.__name__ == "device"
