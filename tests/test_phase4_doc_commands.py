from pathlib import Path


def test_phase4_doc_contains_training_command():
    text = Path("PHASE_4.md").read_text(encoding="utf-8")
    assert "python train.py" in text
    assert "run_generation" in text
