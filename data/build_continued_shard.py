"""Build exactly one processed continued-pretraining shard.

The builder is intentionally separate from training:

    raw 318MB/local shard or one Parquet shard
        -> stream records
        -> clean/filter
        -> persistent exact deduplication
        -> deterministic train/validation split
        -> processed JSONL shard

Only one source shard is materialized on disk at a time. The tokenizer and
model are not touched here; tokenization happens during training using the
existing project tokenizer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.continued_pretraining_stream import (
    ExactDedupStore,
    iter_local_texts,
    iter_training_texts,
)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Build one processed continued-pretraining shard.")
    parser.add_argument("--input", required=True, type=Path, help="One local source file/directory.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for the processed shard.")
    parser.add_argument(
        "--dedup-db",
        type=Path,
        default=None,
        help="Persistent SQLite dedup database. Defaults to <output-dir>/dedup.sqlite3.",
    )
    parser.add_argument("--validation-mod", type=int, default=20)
    parser.add_argument("--parquet-batch-size", type=int, default=4096)
    return parser.parse_args(args)


def _write_split(source, output_path: Path, dedup: ExactDedupStore, split: str, validation_mod: int):
    count = 0
    chars = 0
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for text in iter_training_texts(
            iter_local_texts(source),
            dedup=dedup,
            split=split,
            validation_mod=validation_mod,
        ):
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1
            chars += len(text)
            if count % 1000 == 0:
                dedup.commit()
    temporary.replace(output_path)
    return count, chars


def main(args=None):
    options = parse_args(args)
    if not options.input.exists():
        raise FileNotFoundError(f"Input does not exist: {options.input}")
    if options.validation_mod < 2:
        raise ValueError("validation-mod must be >= 2")
    if options.parquet_batch_size <= 0:
        raise ValueError("parquet-batch-size must be positive")

    options.output_dir.mkdir(parents=True, exist_ok=True)
    dedup_path = options.dedup_db or (options.output_dir / "dedup.sqlite3")
    dedup = ExactDedupStore(dedup_path)

    train_path = options.output_dir / "train.jsonl"
    validation_path = options.output_dir / "validation.jsonl"
    manifest_path = options.output_dir / "manifest.json"

    try:
        train_count, train_chars = _write_split(
            options.input, train_path, dedup, "train", options.validation_mod
        )
        validation_count, validation_chars = _write_split(
            options.input, validation_path, dedup, "validation", options.validation_mod
        )
        dedup.commit()
    finally:
        dedup.close()

    manifest = {
        "source": str(options.input),
        "train": {"documents": train_count, "characters": train_chars},
        "validation": {"documents": validation_count, "characters": validation_chars},
        "validation_mod": options.validation_mod,
        "format": "jsonl",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Processed shard complete")
    print(f"  source: {options.input}")
    print(f"  train: {train_count:,} documents / {train_chars:,} chars")
    print(f"  validation: {validation_count:,} documents / {validation_chars:,} chars")
    print(f"  output: {options.output_dir}")


if __name__ == "__main__":
    main()
