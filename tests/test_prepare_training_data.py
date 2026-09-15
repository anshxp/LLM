from pathlib import Path

import pytest

import data.prepare_training_data as prepare_module


def test_load_token_ids_requires_trained_tokenizer(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.txt"
    tokenizer = tmp_path / "tokenizer.json"
    corpus.write_text("medical text", encoding="utf-8")

    monkeypatch.setattr(prepare_module, "CORPUS_FILE", corpus)
    monkeypatch.setattr(prepare_module, "TOKENIZER_FILE", tokenizer)

    with pytest.raises(FileNotFoundError, match="Tokenizer file not found"):
        prepare_module.load_token_ids()
