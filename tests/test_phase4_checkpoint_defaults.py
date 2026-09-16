import inspect

from training.checkpoint import save_checkpoint


def test_checkpoint_progress_defaults_are_compatible():
    signature = inspect.signature(save_checkpoint)
    assert signature.parameters["epoch"].default == 0
    assert signature.parameters["batch_index"].default == 0
