from pathlib import Path
from typing import Iterator


SUPPORTED_TEXT_EXTENSIONS = {".txt", ".text", ".md"}
SUPPORTED_BINARY_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = SUPPORTED_TEXT_EXTENSIONS | SUPPORTED_BINARY_EXTENSIONS


def _validate_data_dir(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {data_dir}")
    return data_dir


def iter_text_files(data_dir: str | Path) -> Iterator[tuple[str, str]]:
    """Yield supported text files recursively, one file at a time."""
    data_dir = _validate_data_dir(data_dir)

    for file_path in sorted(data_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_TEXT_EXTENSIONS:
            continue

        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"Warning: could not read {file_path}: {exc}")
            continue

        if text.strip():
            yield str(file_path), text


def iter_supported_files(data_dir: str | Path) -> Iterator[Path]:
    """Yield supported corpus source files recursively in stable order."""
    data_dir = _validate_data_dir(data_dir)

    for file_path in sorted(data_dir.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield file_path
