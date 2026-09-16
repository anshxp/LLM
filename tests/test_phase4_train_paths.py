from train import DEFAULT_CHECKPOINT_DIR


def test_default_checkpoint_directory_is_checkpoints():
    assert str(DEFAULT_CHECKPOINT_DIR) == "checkpoints"
