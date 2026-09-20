import json

import torch

from data.instruction_dataset import InstructionDataset, format_example, load_jsonl


def record(response="The heart pumps blood."):
    return {
        "instruction": "Explain the term simply.",
        "input": "heart",
        "response": response,
        "category": "simplification",
    }


def test_format_example_has_explicit_response_boundary():
    prompt, response = format_example(record())
    assert "### Instruction:" in prompt
    assert "### Input:" in prompt
    assert prompt.endswith("### Response:\n")
    assert response == "The heart pumps blood."


def test_dataset_masks_prompt_targets(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(record()) + "\n", encoding="utf-8")

    dataset = InstructionDataset(load_jsonl(path), context_length=128)
    input_ids, labels = dataset[0]

    assert input_ids.dtype == torch.long
    assert labels.dtype == torch.long
    assert input_ids.shape == labels.shape
    assert (labels == -100).any()
    assert (labels != -100).any()


def test_load_jsonl_rejects_duplicates(tmp_path):
    path = tmp_path / "data.jsonl"
    text = json.dumps(record()) + "\n" + json.dumps(record()) + "\n"
    path.write_text(text, encoding="utf-8")

    try:
        load_jsonl(path)
    except ValueError as exc:
        assert "Duplicate" in str(exc)
    else:
        raise AssertionError("duplicate records should be rejected")
