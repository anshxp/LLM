from pathlib import Path


def test_checkpoint_paths_are_path_objects():
    path = Path("checkpoints/model_epoch_1.pt")
    assert path.suffix == ".pt"
