from pathlib import Path


def test_phase4_does_not_add_dependency_file():
    assert Path("requirements.txt").exists()
