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
        max_tokens = self.context_length + 1
        if max_tokens < 2:
            raise ValueError("context_length must be at least 1")
        if not response_ids:
            raise ValueError("Instruction response must contain at least one token")

        # Preserve the response when possible. If the response itself is longer
        # than context, retain its newest tokens and keep EOS as the final token.
        if len(response_ids) >= max_tokens:
            response_ids = response_ids[-max_tokens:]
            prompt_ids = []
        else:
            max_prompt = max_tokens - len(response_ids)
            prompt_ids = prompt_ids[-max_prompt:]
        token_ids = prompt_ids + response_ids

        if len(token_ids) < 2:
            raise ValueError("Instruction example must contain at least two tokens")

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


def collate_instruction_batch(batch):
    """Pad variable-length instruction examples for batched training.

    Input padding uses token id 0. Label padding uses -100 so padded positions
    are ignored by cross-entropy. Because padding is appended after each example,
    it cannot leak information into later non-padding response tokens.
    """
    if not batch:
        raise ValueError("Cannot collate an empty instruction batch")

    max_length = max(inputs.size(0) for inputs, _ in batch)
    input_ids = torch.zeros((len(batch), max_length), dtype=torch.long)
    target_ids = torch.full(
        (len(batch), max_length), -100, dtype=torch.long
    )

    for index, (inputs, labels) in enumerate(batch):
        length = inputs.size(0)
        input_ids[index, :length] = inputs
        target_ids[index, :length] = labels

    return input_ids, target_ids
