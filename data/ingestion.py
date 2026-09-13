from pathlib import Path
from typing import Iterator


SUPPORTED_TEXT_EXTENSIONS = {
    ".txt",
    ".text",
    ".md",
}


def iter_text_files(data_dir: str | Path) -> Iterator[tuple[str, str]]:
    """
    Lazily iterate through text files in a directory.

    Yields:
        (file_path, text)

    Files are read one at a time so the entire corpus does not
    need to fit into RAM.
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    if not data_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {data_dir}")

    for file_path in data_dir.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_TEXT_EXTENSIONS:
            continue

        try:
            text = file_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            print(f"Warning: could not read {file_path}: {exc}")
            continue

        if not text.strip():
            continue

        yield str(file_path), text