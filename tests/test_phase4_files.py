from pathlib import Path


def test_phase4_files_exist():
    required = [
        Path("PHASE_4.md"),
        Path("inference/run_generation.py"),
        Path("tools/model_info.py"),
    ]
    assert all(path.exists() for path in required)
