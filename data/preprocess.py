from pathlib import Path

from data.cleaner import clean_text
from data.filters import passes_basic_filters
from data.ingestion import iter_text_files


def preprocess_directory(
    input_dir: str | Path,
    output_dir: str | Path,
) -> tuple[int, int]:
    """
    Process text documents from input_dir and write accepted
    documents to output_dir.

    Returns:
        (processed_count, rejected_count)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    rejected_count = 0

    for file_path, raw_text in iter_text_files(input_dir):
        cleaned_text = clean_text(raw_text)

        if not passes_basic_filters(cleaned_text):
            rejected_count += 1
            continue

        source_path = Path(file_path)
        relative_path = source_path.relative_to(input_dir)

        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text(
            cleaned_text,
            encoding="utf-8",
        )

        processed_count += 1

    return processed_count, rejected_count