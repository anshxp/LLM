from pathlib import Path

from data.preprocess import preprocess_directory


def test_preprocess_directory(tmp_path: Path):
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"

    input_dir.mkdir()

    valid_text = (
        "Pneumonia is an infection of the lungs. "
        "It may be caused by bacteria, viruses, or other microorganisms. "
        "Clinical presentation can vary depending on the underlying cause. "
        "Patients may develop fever, cough, shortness of breath, chest pain, "
        "and fatigue. Diagnosis commonly involves clinical assessment, "
        "laboratory testing, and imaging when appropriate."
    )

    invalid_text = "Too short."

    (input_dir / "valid.txt").write_text(
        valid_text,
        encoding="utf-8",
    )

    (input_dir / "invalid.txt").write_text(
        invalid_text,
        encoding="utf-8",
    )

    processed, rejected = preprocess_directory(
        input_dir,
        output_dir,
    )

    assert processed == 1
    assert rejected == 1

    output_file = output_dir / "valid.txt"

    assert output_file.exists()
    assert output_file.read_text(encoding="utf-8") == valid_text