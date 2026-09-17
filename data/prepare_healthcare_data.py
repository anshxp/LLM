from pathlib import Path

from data.healthcare_dataset import load_jsonl, split_records, write_jsonl
from data.healthcare_format import format_records

SOURCE_FILE = Path("data/healthcare_examples.jsonl")
PROCESSED_DIR = Path("data/processed")
SPLIT_FILES = {
    "train": PROCESSED_DIR / "healthcare_train.txt",
    "validation": PROCESSED_DIR / "healthcare_validation.txt",
    "test": PROCESSED_DIR / "healthcare_test.txt",
}
SPLIT_JSONL_FILES = {
    "train": PROCESSED_DIR / "healthcare_train.jsonl",
    "validation": PROCESSED_DIR / "healthcare_validation.jsonl",
    "test": PROCESSED_DIR / "healthcare_test.jsonl",
}


def prepare_healthcare_data(source_file: Path = SOURCE_FILE):
    """Validate, split, and format the healthcare corpus for language-model training."""
    records = load_jsonl(source_file)
    splits = split_records(records)

    for split, split_records_list in splits.items():
        write_jsonl(split_records_list, SPLIT_JSONL_FILES[split])
        text = format_records(split_records_list)
        SPLIT_FILES[split].parent.mkdir(parents=True, exist_ok=True)
        SPLIT_FILES[split].write_text(text, encoding="utf-8")

    print(f"Healthcare examples: {len(records):,}")
    for split, split_records_list in splits.items():
        print(f"{split}: {len(split_records_list):,} examples -> {SPLIT_FILES[split]}")

    return splits


if __name__ == "__main__":
    prepare_healthcare_data()
