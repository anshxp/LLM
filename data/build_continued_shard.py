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
from typing import Iterator

from data.continued_pretraining_stream import (
    ExactDedupStore,
    iter_local_shards,
    iter_local_texts,
    iter_training_texts,
)

COMPLETE_MARKER = ".complete"


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Build one processed continued-pretraining shard.")
    parser.add_argument("--input", required=True, type=Path, help="One local source file or directory.")
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


def _source_texts(source: Path, parquet_batch_size: int) -> Iterator[str]:
    for _, path in iter_local_shards(source):
        if path.suffix.lower() == ".parquet":
            from data.continued_pretraining_stream import iter_parquet_records, record_to_text

            for row in iter_parquet_records(path, batch_size=parquet_batch_size):
                text = record_to_text(row)
                if text:
                    yield text
        else:
            yield from iter_local_texts(path)


def _write_split(
    source: Path,
    output_path: Path,
    dedup: ExactDedupStore,
    split: str,
    validation_mod: int,
    parquet_batch_size: int,
):
    count = 0
    chars = 0
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for text in iter_training_texts(
            _source_texts(source, parquet_batch_size),
            dedup=dedup,
            split=split,
            validation_mod=validation_mod,
        ):
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1
            chars += len(text)
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
    complete_marker = options.output_dir / COMPLETE_MARKER
    if complete_marker.exists():
        print(f"Processed shard already complete: {options.output_dir}")
        return

    dedup_path = options.dedup_db or (options.output_dir / "dedup.sqlite3")
    dedup = ExactDedupStore(dedup_path)
    # Keep the whole shard's dedup transaction open. If the process fails,
    # SQLite rolls back the uncommitted transaction, so a rebuild cannot lose
    # documents merely because an earlier partial output existed.
    dedup.commit_every = 10**18

    train_path = options.output_dir / "train.jsonl"
    validation_path = options.output_dir / "validation.jsonl"
    manifest_path = options.output_dir / "manifest.json"

    try:
        train_count, train_chars = _write_split(
            options.input,
            train_path,
            dedup,
            "train",
            options.validation_mod,
            options.parquet_batch_size,
        )
        validation_count, validation_chars = _write_split(
            options.input,
            validation_path,
            dedup,
            "validation",
            options.validation_mod,
            options.parquet_batch_size,
        )

        manifest = {
            "source": str(options.input),
            "train": {"documents": train_count, "characters": train_chars},
            "validation": {"documents": validation_count, "characters": validation_chars},
            "validation_mod": options.validation_mod,
            "format": "jsonl",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        dedup.commit()
        complete_marker.write_text("complete\n", encoding="utf-8")
    finally:
        dedup.close()

    print("Processed shard complete")
    print(f"  source: {options.input}")
    print(f"  train: {train_count:,} documents / {train_chars:,} chars")
    print(f"  validation: {validation_count:,} documents / {validation_chars:,} chars")
    print(f"  output: {options.output_dir}")


if __name__ == "__main__":
    main()
