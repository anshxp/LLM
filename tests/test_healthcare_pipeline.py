from pathlib import Path

from data.healthcare_dataset import split_records
from data.healthcare_format import format_records


def make_records(count=10):
    return [
        {
            "instruction": f"Explain term {index}",
            "input": f"Term {index}",
            "response": f"Simple explanation {index}",
            "category": "terminology" if index % 2 == 0 else "health_information",
        }
        for index in range(count)
    ]


def test_healthcare_records_split_and_format():
    train, validation, test = split_records(make_records())

    assert len(train) == 8
    assert len(validation) == 1
    assert len(test) == 1

    formatted = format_records(train)
    assert "### Instruction" in formatted
    assert "### Input" in formatted
    assert "### Response" in formatted
    assert "Simple explanation 0" in formatted


def test_healthcare_split_is_disjoint():
    train, validation, test = split_records(make_records())

    def key(record):
        return (record["instruction"], record["input"], record["response"])

    train_keys = {key(record) for record in train}
    validation_keys = {key(record) for record in validation}
    test_keys = {key(record) for record in test}

    assert train_keys.isdisjoint(validation_keys)
    assert train_keys.isdisjoint(test_keys)
    assert validation_keys.isdisjoint(test_keys)
