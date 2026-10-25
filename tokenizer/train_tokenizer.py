"""Train a new BPE tokenizer from Common Pile PubMed Parquet shards."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from datetime import datetime, timezone

import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, normalizers


def list_shards(data_dir: Path):
    files = sorted(data_dir.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No part-*.parquet shards found in {data_dir}")
    return files


def count_rows(
    files,
    text_column: str,
    max_rows: int | None,
    log_every_shards: int = 50,
) -> int:
    """Count candidate rows from Parquet metadata and log scan progress."""
    total = 0
    started = time.monotonic()
    shard_count = len(files)
    print(
        f"[metadata] Starting scan of {shard_count:,} Parquet shards "
        f"(progress every {log_every_shards} shards).",
        flush=True,
    )

    for index, file in enumerate(files, start=1):
        # Log before opening the file so a slow or blocked Drive read is visible.
        if index == 1 or (index - 1) % log_every_shards == 0:
            elapsed = time.monotonic() - started
            print(
                f"[metadata] Opening shard {index:,}/{shard_count:,} "
                f"| Rows counted: {total:,} | Elapsed: {elapsed:.1f}s "
                f"| Current file: {file.name}",
                flush=True,
            )
        try:
            parquet = pq.ParquetFile(file)
            if text_column not in parquet.schema_arrow.names:
                raise ValueError(
                    f"Column {text_column!r} not found in shard {file}"
                )
            total += parquet.metadata.num_rows
        except Exception as exc:
            elapsed = time.monotonic() - started
            raise RuntimeError(
                f"Metadata scan failed at shard {index}/{shard_count}: {file} "
                f"after {elapsed:.1f}s; rows counted so far: {total:,}. "
                "Check Google Drive mount/access and shard integrity."
            ) from exc

        if max_rows is not None and total >= max_rows:
            elapsed = time.monotonic() - started
            print(
                f"[metadata] Reached --max-rows={max_rows:,} at shard "
                f"{index:,}/{shard_count:,} in {elapsed:.1f}s.",
                flush=True,
            )
            return max_rows

    elapsed = time.monotonic() - started
    print(
        f"[metadata] Scan complete: {shard_count:,} shards, "
        f"{total:,} rows, {elapsed:.1f}s.",
        flush=True,
    )
    return total


def iter_texts(files, text_column: str, max_rows: int | None, log_every: int):
    seen = 0
    started = time.monotonic()
    last_log = started

    for file_index, file in enumerate(files, start=1):
        parquet = pq.ParquetFile(file)
        for batch in parquet.iter_batches(columns=[text_column], batch_size=256):
            for value in batch.column(0).to_pylist():
                if not isinstance(value, str) or not value.strip():
                    continue

                yield value
                seen += 1

                now = time.monotonic()
                if seen % log_every == 0 or now - last_log >= 30:
                    elapsed = now - started
                    rate = seen / elapsed if elapsed else 0
                    print(
                        f"[input] Documents yielded: {seen:,} | "
                        f"Elapsed: {elapsed:.1f}s | Rate: {rate:.1f} docs/s | "
                        f"Current shard: {file_index}/{len(files)}",
                        flush=True,
                    )
                    last_log = now

                if max_rows is not None and seen >= max_rows:
                    print(f"[input] Reached --max-rows limit ({max_rows:,}).", flush=True)
                    return

    elapsed = time.monotonic() - started
    print(
        f"[input] Finished reading {seen:,} nonempty documents in {elapsed:.1f}s.",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("/content/drive/MyDrive/LLM/common_pile_pubmed"))
    p.add_argument("--output-dir", type=Path, default=Path("/content/drive/MyDrive/LLM/tokenizer_bpe32k"))
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test limit; omit for all available shards")
    p.add_argument("--text-column", default="text")
    p.add_argument("--log-every", type=int, default=1000, help="Print input progress every N nonempty documents")
    args = p.parse_args()

    if args.vocab_size < 256:
        p.error("--vocab-size must be at least 256")
    if args.min_frequency < 1:
        p.error("--min-frequency must be positive")
    if args.max_rows is not None and args.max_rows < 1:
        p.error("--max-rows must be positive")
    if args.log_every < 1:
        p.error("--log-every must be positive")

    files = list_shards(args.data_dir)
    total_rows = count_rows(files, args.text_column, args.max_rows)
    if total_rows == 0:
        raise ValueError(f"No rows found in {args.data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / "tokenizer.json"
    if output_file.exists():
        raise FileExistsError(
            f"{output_file} already exists; choose another output directory "
            "or move the old tokenizer first."
        )

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SYSTEM]", "[USER]", "[ASSISTANT]"]
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"Training a new BPE tokenizer from: {args.data_dir}", flush=True)
    print(f"Parquet shards: {len(files):,}", flush=True)
    print(f"Candidate rows from metadata: {total_rows:,}", flush=True)
    print(f"Vocab target: {args.vocab_size:,}; minimum frequency: {args.min_frequency}", flush=True)
    print(f"Input progress logs every {args.log_every:,} nonempty documents.", flush=True)
    print("The corpus is streamed; it is not copied into RAM.", flush=True)

    training_started = time.monotonic()
    tokenizer.train_from_iterator(
        iter_texts(files, args.text_column, args.max_rows, args.log_every),
        trainer=trainer,
        length=total_rows,
    )
    training_seconds = time.monotonic() - training_started
    tokenizer.save(str(output_file))

    probes = [
        "pneumothorax", "pleural effusion", "atelectasis", "cardiomegaly",
        "pulmonary edema", "consolidation", "ground-glass opacity",
        "No focal air-space consolidation or pleural effusion.",
    ]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "BPE",
        "normalizer": "NFKC",
        "pre_tokenizer": "ByteLevel",
        "target_vocab_size": args.vocab_size,
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": special_tokens,
        "training_data_dir": str(args.data_dir),
        "text_column": args.text_column,
        "candidate_rows_from_metadata": total_rows,
        "max_rows_limit": args.max_rows,
        "training_seconds": round(training_seconds, 3),
        "probe_tokenization": {
            text: {"tokens": tokenizer.encode(text).tokens, "ids": tokenizer.encode(text).ids}
            for text in probes
        },
        "note": "General PubMed corpus; clinical terms are probes, not a corpus filter.",
    }
    report_path = args.output_dir / "tokenizer_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved tokenizer: {output_file}", flush=True)
    print(f"Actual vocabulary size: {tokenizer.get_vocab_size():,}", flush=True)
    print(f"Training elapsed: {training_seconds:.1f}s", flush=True)
    print(f"Saved report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
