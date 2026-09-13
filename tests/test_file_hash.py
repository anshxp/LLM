from pathlib import Path

from data.file_hash import calculate_sha256


def test_calculate_sha256(tmp_path: Path):
    test_file = tmp_path / "test.txt"
    test_file.write_bytes(b"hello healthcare LLM")

    first_hash = calculate_sha256(test_file)
    second_hash = calculate_sha256(test_file)

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_different_files_have_different_hashes(tmp_path: Path):
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"

    file_a.write_bytes(b"document A")
    file_b.write_bytes(b"document B")

    assert calculate_sha256(file_a) != calculate_sha256(file_b)