from pathlib import Path

import pytest

import data.prepare_training_data as prepare_module


def test_load_token_ids_requires_trained_tokenizer(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.txt"
    tokenizer = tmp_path / "tokenizer.json"
    corpus.write_text("medical text " * 30, encoding="utf-8")

    monkeypatch.setattr(prepare_module, "TOKENIZER_FILE", tokenizer)

    with pytest.raises(FileNotFoundError, match="Tokenizer file not found"):
        prepare_module.load_token_ids()


def test_token_store_path_is_dataset_specific(tmp_path):
    assert prepare_module.token_store_path(
        "base", "train", tmp_path
    ) == tmp_path / "base" / "train.tokens"


def test_prepare_dataset_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="Unknown dataset"):
        prepare_module.prepare_dataset("missing")
