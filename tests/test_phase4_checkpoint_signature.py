import inspect

from training.checkpoint import save_checkpoint


def test_checkpoint_accepts_progress_metadata():
    parameters = inspect.signature(save_checkpoint).parameters
    assert "epoch" in parameters
    assert "batch_index" in parameters
