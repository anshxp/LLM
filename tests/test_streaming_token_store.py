from pathlib import Path

import numpy as np

import data.prepare_training_data as prepare_module
from data.memmap_dataset import MemmapLanguageModelDataset


class FakeTokenizer:
    def __len__(self):
        return 10000

    def encode(self, text):
        return [len(text) % 997, 1, 2, 3]


def test_prepare_memmap_token_store_is_bounded_and_atomic(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcdefghij" * 20, encoding="utf-8")
    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer_file.write_text("stub", encoding="utf-8")
    output = tmp_path / "train.tokens"

    monkeypatch.setattr(
        prepare_module.Tokenizer,
        "from_file",
        classmethod(lambda cls, path: FakeTokenizer()),
    )

    result = prepare_module.build_memmap_token_store(
        corpus,
        output,
        tokenizer_file=tokenizer_file,
        chunk_chars=10,
    )

    assert result["tokens"] == 80
    assert output.exists()
    assert not list(tmp_path.glob("*.tmp"))

    dataset = MemmapLanguageModelDataset(output, context_length=4)
    assert len(dataset) > 0
    inputs, targets = dataset[0]
    assert inputs.shape == targets.shape == (4,)
    assert inputs.dtype == targets.dtype


def test_prepare_dataset_does_not_process_other_datasets(tmp_path, monkeypatch):
    base_train = tmp_path / "train.txt"
    base_val = tmp_path / "validation.txt"
    base_test = tmp_path / "test.txt"
    for path in (base_train, base_val, base_test):
        path.write_text("abcdefghij" * 30, encoding="utf-8")

    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer_file.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(
        prepare_module,
        "SPLIT_FILES",
        {
            "base": {
                "train": base_train,
                "validation": base_val,
                "test": base_test,
            },
            "healthcare": {
                "train": tmp_path / "missing_healthcare.txt",
                "validation": tmp_path / "missing_healthcare_val.txt",
                "test": tmp_path / "missing_healthcare_test.txt",
            },
        },
    )
    monkeypatch.setattr(
        prepare_module.Tokenizer,
        "from_file",
        classmethod(lambda cls, path: FakeTokenizer()),
    )

    results = prepare_module.prepare_dataset(
        "base",
        tokenizer_file=tokenizer_file,
        output_root=tmp_path / "tokens",
        chunk_chars=16,
    )
    assert {Path(item["source"]).name for item in results} == {
        "train.txt",
        "validation.txt",
        "test.txt",
    }
