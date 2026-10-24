"""Benchmark BPE vocabulary sizes on a sampled train/evaluation set from Parquet shards.

The sampler chooses shards spread across the source manifest, then reservoir-samples
rows from those shards. It does not load the whole corpus into RAM.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, normalizers


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SYSTEM]", "[USER]", "[ASSISTANT]"]


def sample_from_shard(path: Path, text_column: str, count: int, rng: random.Random):
    """Reservoir sample up to count nonempty text rows from one shard."""
    reservoir: list[str] = []
    seen = 0
    parquet = pq.ParquetFile(path)
    if text_column not in parquet.schema_arrow.names:
        raise ValueError(f"Column {text_column!r} not found in {path}")

    for batch in parquet.iter_batches(columns=[text_column], batch_size=256):
        for value in batch.column(0).to_pylist():
            if not isinstance(value, str) or not value.strip():
                continue
            seen += 1
            if len(reservoir) < count:
                reservoir.append(value)
            else:
                j = rng.randrange(seen)
                if j < count:
                    reservoir[j] = value
    return reservoir, seen


def build_samples(data_dir: Path, out_dir: Path, text_column: str,
                  train_rows: int, eval_rows: int, seed: int):
    train_path = out_dir / "train_sample.parquet"
    eval_path = out_dir / "eval_sample.parquet"
    sample_meta = out_dir / "sample_manifest.json"

    if train_path.exists() and eval_path.exists() and sample_meta.exists():
        meta = json.loads(sample_meta.read_text(encoding="utf-8"))
        if meta.get("train_rows_requested") == train_rows and meta.get("eval_rows_requested") == eval_rows:
            print(f"Reusing cached samples: {train_path} and {eval_path}", flush=True)
            return train_path, eval_path, meta

    files = sorted(data_dir.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No part-*.parquet shards found in {data_dir}")

    # Spread sampling across the full corpus. More shards improve representativeness,
    # while bounding the amount of Drive I/O for a pilot benchmark.
    selected_count = min(len(files), max(64, min(512, (train_rows + eval_rows + 49) // 50)))
    indices = sorted({
        round(i * (len(files) - 1) / max(1, selected_count - 1))
        for i in range(selected_count)
    })
    selected = [files[i] for i in indices]
    print(f"Sampling from {len(selected)} evenly spaced shards out of {len(files)}.", flush=True)

    # Allocate a proportional quota per selected shard; use reservoir sampling
    # within each shard so selected rows are not restricted to the start of each file.
    total_needed = train_rows + eval_rows
    per_shard = (total_needed + len(selected) - 1) // len(selected)
    rng = random.Random(seed)
    documents: list[str] = []

    for n, path in enumerate(selected, start=1):
        remaining = total_needed - len(documents)
        if remaining <= 0:
            break
        quota = min(per_shard, remaining)
        sampled, nonempty = sample_from_shard(path, text_column, quota, rng)
        documents.extend(sampled)
        if n % 16 == 0 or n == len(selected):
            print(
                f"[sampling] Shards visited: {n}/{len(selected)} | "
                f"Documents collected: {len(documents):,}/{total_needed:,}",
                flush=True,
            )

    if len(documents) < total_needed:
        raise RuntimeError(
            f"Only collected {len(documents)} nonempty documents; requested {total_needed}. "
            "Increase selected shard count or lower sample sizes."
        )

    rng.shuffle(documents)
    train_docs = documents[:train_rows]
    eval_docs = documents[train_rows:train_rows + eval_rows]
    pq.write_table(pa.table({"text": train_docs}), train_path, compression="zstd")
    pq.write_table(pa.table({"text": eval_docs}), eval_path, compression="zstd")

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(data_dir),
        "text_column": text_column,
        "sampling_method": "Evenly spaced source shards, reservoir sample within each selected shard, then shuffle",
        "seed": seed,
        "selected_shards": len(selected),
        "train_rows_requested": train_rows,
        "train_rows_actual": len(train_docs),
        "eval_rows_requested": eval_rows,
        "eval_rows_actual": len(eval_docs),
    }
    sample_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {len(train_docs):,} training docs and {len(eval_docs):,} evaluation docs.", flush=True)
    return train_path, eval_path, meta


def read_texts(path: Path):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["text"], batch_size=256):
        for value in batch.column(0).to_pylist():
            if isinstance(value, str) and value.strip():
                yield value


def make_tokenizer(train_path: Path, vocab_size: int, min_frequency: int):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    rows = pq.ParquetFile(train_path).metadata.num_rows
    tokenizer.train_from_iterator(read_texts(train_path), trainer=trainer, length=rows)
    return tokenizer


def evaluate(tokenizer: Tokenizer, eval_path: Path, log_every: int = 100):
    docs = 0
    chars = 0
    tokens = 0
    started = time.monotonic()
    last_log = started
    for text in read_texts(eval_path):
        encoded = tokenizer.encode(text)
        docs += 1
        chars += len(text)
        tokens += len(encoded.ids)
        now = time.monotonic()
        if docs % log_every == 0 or now - last_log >= 30:
            print(f"[eval] Documents: {docs:,} | Tokens: {tokens:,}", flush=True)
            last_log = now
    elapsed = time.monotonic() - started
    return {
        "eval_documents": docs,
        "eval_characters": chars,
        "eval_tokens": tokens,
        "tokens_per_character": tokens / chars if chars else None,
        "characters_per_token": chars / tokens if tokens else None,
        "eval_seconds": round(elapsed, 3),
        "eval_documents_per_second": docs / elapsed if elapsed else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--vocab-sizes", default="32000,50000,64000,100000,150000,200000,300000")
    p.add_argument("--train-rows", type=int, default=10000)
    p.add_argument("--eval-rows", type=int, default=1000)
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--text-column", default="text")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-log-every", type=int, default=100)
    args = p.parse_args()

    sizes = []
    for item in args.vocab_sizes.split(","):
        size = int(item.strip())
        if size < 256:
            p.error("Every vocabulary size must be at least 256.")
        if size not in sizes:
            sizes.append(size)
    if not sizes:
        p.error("--vocab-sizes must contain at least one size.")
    if args.train_rows < 1 or args.eval_rows < 1:
        p.error("--train-rows and --eval-rows must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path, eval_path, sample_meta = build_samples(
        args.data_dir, args.output_dir, args.text_column,
        args.train_rows, args.eval_rows, args.seed,
    )

    results = []
    summary_path = args.output_dir / "benchmark_results.json"
    csv_path = args.output_dir / "benchmark_results.csv"

    for size in sizes:
        name = f"{size // 1000}k" if size % 1000 == 0 else str(size)
        run_dir = args.output_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_path = run_dir / "tokenizer.json"
        report_path = run_dir / "report.json"

        if report_path.exists() and tokenizer_path.exists():
            print(f"\nSkipping {size:,}; completed report already exists.", flush=True)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            results.append(report)
            continue

        print("\n" + "=" * 70, flush=True)
        print(f"START vocabulary target {size:,} at {datetime.now(timezone.utc).isoformat()}", flush=True)
        print(f"Training docs: {args.train_rows:,}; evaluation docs: {args.eval_rows:,}", flush=True)
        print("=" * 70, flush=True)

        started = time.monotonic()
        tokenizer = make_tokenizer(train_path, size, args.min_frequency)
        train_seconds = time.monotonic() - started
        actual_size = tokenizer.get_vocab_size()
        tokenizer.save(str(tokenizer_path))

        eval_metrics = evaluate(tokenizer, eval_path, args.eval_log_every)
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "algorithm": "BPE",
            "target_vocab_size": size,
            "actual_vocab_size": actual_size,
            "min_frequency": args.min_frequency,
            "training_documents": args.train_rows,
            "evaluation_documents": eval_metrics["eval_documents"],
            "training_seconds": round(train_seconds, 3),
            "training_documents_per_second": args.train_rows / train_seconds if train_seconds else None,
            "tokenizer_file": str(tokenizer_path),
            "train_sample": str(train_path),
            "eval_sample": str(eval_path),
            **eval_metrics,
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        results.append(report)

        # Persist after every vocabulary size so completed work survives a runtime interruption.
        summary_path.write_text(json.dumps({"sample": sample_meta, "results": results}, indent=2), encoding="utf-8")
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            fields = sorted({key for row in results for key in row.keys()})
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)

        print(
            f"FINISHED {size:,}: actual vocab={actual_size:,} | "
            f"train={train_seconds:.1f}s | "
            f"eval tokens/char={eval_metrics['tokens_per_character']:.5f} | "
            f"eval chars/token={eval_metrics['characters_per_token']:.3f}",
            flush=True,
        )
        print(f"Saved report: {report_path}", flush=True)

    print("\nBenchmark complete.")
    print(f"JSON results: {summary_path}")
    print(f"CSV results: {csv_path}")


if __name__ == "__main__":
    main()
