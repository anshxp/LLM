from training.checkpoint import save_checkpoint


def test_save_checkpoint_is_callable():
    assert callable(save_checkpoint)
