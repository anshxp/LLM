"""Dataset utilities for supervised instruction fine-tuning."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from data.healthcare_schema import validate_example
from data.tokenizer import Tokenizer


DEFAULT_TOKENIZER = Path("data/processed/tokenizer.json")


def format_example(record):
    """Format an instruction example into the training prompt template."""
    record = validate_example(record)
    instruction = record["instruction"]
    user_input = record["input"]
    response = record["response"]
    prompt = (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Input:\n"
        f"{user_input}\n\n"
        "### Response:\n"
    )
    return prompt, response


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Instruction dataset not found: {path}")

    records = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = validate_example(json.loads(line))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid instruction record at line {line_number}: {exc}") from exc
        key = tuple(record[field] for field in ("instruction", "input", "response"))
        if key in seen:
            raise ValueError(f"Duplicate instruction record at line {line_number}")
        seen.add(key)
        records.append(record)

    if not records:
        raise ValueError("Instruction dataset contains no examples")
    return records


class InstructionDataset(Dataset):
    """Tokenized supervised examples with loss applied only to response tokens."""

    def __init__(self, records, context_length, tokenizer_path=DEFAULT_TOKENIZER):
        self.records = list(records)
        self.context_length = context_length
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.examples = [self._encode(record) for record in self.records]

    def _encode(self, record):
        prompt, response = format_example(record)
        prompt_ids = self.tokenizer.encode(prompt, add_bos=True)
        response_ids = self.tokenizer.encode(response, add_eos=True)
        token_ids = prompt_ids + response_ids

        if len(token_ids) < 2:
            raise ValueError("Instruction example must contain at least two tokens")
        if len(token_ids) > self.context_length + 1:
            # Preserve the response and truncate the oldest prompt tokens first.
            max_prompt = max(0, self.context_length + 1 - len(response_ids))
            prompt_ids = prompt_ids[-max_prompt:] if max_prompt else []
            token_ids = prompt_ids + response_ids
        if len(token_ids) > self.context_length + 1:
            raise ValueError("Response is longer than the model context length")

        inputs = torch.tensor(token_ids[:-1], dtype=torch.long)
        labels = torch.tensor(token_ids[1:], dtype=torch.long)

        # Tokens whose target is still part of the prompt do not contribute to loss.
        prompt_target_count = max(0, len(prompt_ids) - 1)
        labels[:prompt_target_count] = -100
        return inputs, labels

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
