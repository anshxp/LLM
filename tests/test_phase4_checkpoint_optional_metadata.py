import inspect

from training.checkpoint import save_checkpoint


def test_progress_metadata_has_defaults():
    sig = inspect.signature(save_checkpoint)
    assert sig.parameters["epoch"].default == 0
    assert sig.parameters["batch_index"].default == 0
