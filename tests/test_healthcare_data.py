import json

import pytest

from data.healthcare_dataset import load_jsonl, split_records, write_jsonl
from data.healthcare_format import format_example
from data.healthcare_schema import validate_example


VALID = {
    "instruction": "Explain a medical term simply.",
    "input": "What is hypertension?",
    "response": "Hypertension means high blood pressure.",
    "category": "terminology",
}


def test_validate_example_normalizes_whitespace():
    record = validate_example({key: f"  {value}  " for key, value in VALID.items()})
    assert record == VALID


def test_validate_example_rejects_missing_fields():
    with pytest.raises(ValueError):
        validate_example({"instruction": "Explain this."})


def test_validate_example_rejects_empty_fields():
    invalid = dict(VALID, response="   ")
    with pytest.raises(ValueError):
        validate_example(invalid)


def test_load_jsonl_rejects_duplicates(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(VALID) + "\n" + json.dumps(VALID) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_jsonl(path)


def test_load_and_split_jsonl(tmp_path):
    records = [dict(VALID, input=f"Question {index}") for index in range(10)]
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    loaded = load_jsonl(path)
    splits = split_records(loaded, validation_ratio=0.2, test_ratio=0.2)

    assert len(splits["train"]) == 6
    assert len(splits["validation"]) == 2
    assert len(splits["test"]) == 2


def test_split_rejects_invalid_ratios():
    with pytest.raises(ValueError):
        split_records([VALID] * 5, validation_ratio=0.8, test_ratio=0.3)


def test_write_jsonl_round_trip(tmp_path):
    path = tmp_path / "nested" / "data.jsonl"
    write_jsonl([VALID], path)
    assert load_jsonl(path) == [VALID]


def test_format_example_contains_instruction_and_response_sections():
    text = format_example(VALID)
    assert "### Instruction" in text
    assert "### Input" in text
    assert "### Response" in text
    assert VALID["response"] in text
