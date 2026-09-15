from train import DEFAULT_EPOCHS, DEFAULT_LOG_EVERY


def test_training_defaults_are_positive():
    assert DEFAULT_EPOCHS > 0
    assert DEFAULT_LOG_EVERY > 0
