def test_phase4_components_import_together():
    from inference.generate import generate
    from tools.model_info import parameter_count
    from train import parse_args
    from training.checkpoint import save_checkpoint
    from training.trainer import train_step

    assert all(callable(item) for item in (generate, parameter_count, parse_args, save_checkpoint, train_step))
