"""Convert healthcare instruction records into causal-LM training text."""

from data.healthcare_schema import validate_example


def format_example(example):
    """Format one validated record into a plain-text instruction/response sample."""
    record = validate_example(example)
    return (
        "### Instruction\n"
        f"{record['instruction']}\n\n"
        "### Input\n"
        f"{record['input']}\n\n"
        "### Response\n"
        f"{record['response']}"
    )


def format_records(records):
    """Format records for causal language-model pretraining/fine-tuning."""
    return "\n\n".join(format_example(record) for record in records)
