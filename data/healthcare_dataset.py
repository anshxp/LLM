"""Utilities for validating and splitting healthcare instruction data."""

import json
from pathlib import Path

from data.healthcare_schema import validate_example


CATEGORIES = {
    "terminology",
    "simplification",
    "summarization",
    "health_information",
}


def load_jsonl(path):
    """Load and validate one JSONL healthcare dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    records = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = validate_example(json.loads(line))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid record at line {line_number}: {exc}") from exc

        if record["category"] not in CATEGORIES:
            raise ValueError(f"Unknown category at line {line_number}: {record['category']}")

        key = tuple(record[field] for field in ("instruction", "input", "response"))
        if key in seen:
            raise ValueError(f"Duplicate record at line {line_number}")
        seen.add(key)
        records.append(record)

    if not records:
        raise ValueError("Dataset contains no examples")
    return records


def split_records(records, validation_ratio=0.1, test_ratio=0.1):
    """Deterministically split records into train, validation, and test sets."""
    if not records:
        raise ValueError("records must not be empty")
    if validation_ratio < 0 or test_ratio < 0 or validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio and test_ratio must be non-negative and sum to less than 1")

    total = len(records)
    test_count = max(1, round(total * test_ratio)) if test_ratio else 0
    validation_count = max(1, round(total * validation_ratio)) if validation_ratio else 0
    if validation_count + test_count >= total:
        raise ValueError("Dataset is too small for the requested split ratios")

    train_end = total - validation_count - test_count
    return {
        "train": records[:train_end],
        "validation": records[train_end:train_end + validation_count],
        "test": records[train_end + validation_count:],
    }


def write_jsonl(records, path):
    """Write validated records as UTF-8 JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(validate_example(record), ensure_ascii=False) + "\n")
