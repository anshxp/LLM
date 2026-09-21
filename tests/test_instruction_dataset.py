import json

import pytest
import torch
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from data.instruction_dataset import (
    InstructionDataset,
    collate_instruction_batch,
    format_example,
    load_jsonl,
)


def record(response="The heart pumps blood."):
    return {
        "instruction": "Explain the term simply.",
        "input": "heart",
        "response": response,
        "category": "simplification",
    }


@pytest.fixture
def tokenizer_path(tmp_path):
    path = tmp_path / "tokenizer.json"
    tokenizer = HFTokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=64,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        min_frequency=1,
    )
    tokenizer.train_from_iterator(
        [
            "### Instruction: Explain the term simply.",
            "### Input: heart",
            "### Response: The heart pumps blood.",
        ],
        trainer=trainer,
    )
    tokenizer.save(str(path))
    return path


def test_format_example_has_explicit_response_boundary():
    prompt, response = format_example(record())
    assert "### Instruction:" in prompt
    assert "### Input:" in prompt
    assert prompt.endswith("### Response:\n")
    assert response == "The heart pumps blood."


def test_dataset_masks_prompt_targets(tokenizer_path):
    dataset = InstructionDataset(
        [record()],
        context_length=128,
        tokenizer_path=tokenizer_path,
    )
    input_ids, labels = dataset[0]

    assert input_ids.dtype == torch.long
    assert labels.dtype == torch.long
    assert input_ids.shape == labels.shape
    assert (labels == -100).any()
    assert (labels != -100).any()


def test_dataset_truncates_long_response_to_context(tokenizer_path):
    long_response = " ".join(["long"] * 256)
    dataset = InstructionDataset(
        [record(long_response)],
        context_length=32,
        tokenizer_path=tokenizer_path,
    )
    input_ids, labels = dataset[0]

    assert input_ids.shape == labels.shape
    assert input_ids.size(0) <= 32
    assert (labels != -100).any()


def test_collate_instruction_batch_pads_labels_with_ignore_index():
    short = (
        torch.tensor([1, 2, 3], dtype=torch.long),
        torch.tensor([-100, 5, 6], dtype=torch.long),
    )
    long = (
        torch.tensor([7, 8, 9, 10], dtype=torch.long),
        torch.tensor([-100, -100, 11, 12], dtype=torch.long),
    )

    input_ids, labels = collate_instruction_batch([short, long])

    assert input_ids.tolist() == [[1, 2, 3, 0], [7, 8, 9, 10]]
    assert labels.tolist() == [[-100, 5, 6, -100], [-100, -100, 11, 12]]


def test_load_jsonl_rejects_duplicates(tmp_path):
    path = tmp_path / "data.jsonl"
    text = json.dumps(record()) + "\n" + json.dumps(record()) + "\n"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate"):
        load_jsonl(path)
