from pathlib import Path

import torch

from data.build_continued_corpus import _assert_allowed_path, _split_for_hash
from data.continued_pretraining_stream import ExactDedupStore, iter_training_texts, record_to_text
from train_continued_pretraining import parse_args


def test_raw_folder_is_rejected():
    try:
        _assert_allowed_path(Path("data/raw/example.txt"))
    except ValueError:
        return
    raise AssertionError("data/raw must never be accepted as a continued-pretraining source")


def test_streaming_cli_defaults(tmp_path):
    args = parse_args(["--local-shard", str(tmp_path)])
    assert args.pretrained_checkpoint == Path("checkpoints/phase7_run/best_model.pt")
    assert args.tokenizer == Path("data/processed/tokenizer.json")
    assert args.checkpoint_dir == Path("checkpoints/continued_pretraining")
    assert args.hf_pattern == "*.parquet"
    assert args.batch_size == 1
    assert args.gradient_accumulation_steps == 4


def test_split_assignment_is_deterministic():
    assert _split_for_hash("0" * 64) == _split_for_hash("0" * 64)
    assert _split_for_hash("f" * 64) == _split_for_hash("f" * 64)


def test_structured_record_to_text():
    text = record_to_text({"question": "What is X?", "answer": "X is Y."})
    assert text == "Question: What is X?\nAnswer: X is Y."


def test_disk_dedup_and_validation_split(tmp_path):
    store = ExactDedupStore(tmp_path / "dedup.sqlite3")
    try:
        texts = ["A sufficiently long document with useful training content."]
        first = list(iter_training_texts(texts, dedup=store, split="train", validation_mod=2))
        second = list(iter_training_texts(texts, dedup=store, split="train", validation_mod=2))
        assert first in ([], texts)
        assert second == []
    finally:
        store.close()


def test_checkpoint_rng_and_extra_state_are_supported(tmp_path):
    from config.model_config import ModelConfig
    from model.llm import LLM
    from training.checkpoint import load_checkpoint, save_checkpoint
    from training.optimizer import create_optimizer

    torch.manual_seed(123)
    model = LLM(ModelConfig())
    optimizer = create_optimizer(model, learning_rate=1e-4, weight_decay=0.0)
    path = tmp_path / "resume.pt"
    expected = torch.get_rng_state().clone()
    save_checkpoint(
        model,
        optimizer,
        7,
        path,
        epoch=1,
        batch_index=4,
        extra_state={"shard_index": 3, "completed_shards": 3},
    )
    torch.manual_seed(999)
    state = load_checkpoint(model, optimizer, path, map_location="cpu")
    assert torch.equal(torch.get_rng_state(), expected)
    assert state.get("shard_index") == 3
    assert state.get("completed_shards") == 3
