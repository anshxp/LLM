"""Schema and validation helpers for healthcare instruction examples."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HealthcareExample:
    """One patient-facing healthcare instruction example."""

    instruction: str
    input: str
    response: str
    category: str

    def to_record(self):
        return {
            "instruction": self.instruction.strip(),
            "input": self.input.strip(),
            "response": self.response.strip(),
            "category": self.category.strip(),
        }


def validate_example(example):
    """Validate a healthcare example and return a normalized record."""
    if not isinstance(example, dict):
        raise TypeError("example must be a dictionary")

    required = ("instruction", "input", "response", "category")
    missing = [field for field in required if field not in example]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    values = {field: example[field] for field in required}
    for field, value in values.items():
        if not isinstance(value, str):
            raise TypeError(f"{field} must be a string")
        if not value.strip():
            raise ValueError(f"{field} must not be empty")

    return {field: values[field].strip() for field in required}
