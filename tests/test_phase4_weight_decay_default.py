from train import DEFAULT_LEARNING_RATE, DEFAULT_WEIGHT_DECAY


def test_optimizer_defaults_are_positive():
    assert DEFAULT_LEARNING_RATE > 0
    assert DEFAULT_WEIGHT_DECAY >= 0
