"""Benchmark BPE vocabulary sizes on the same Common Pile PubMed sample.

This is a comparative experiment, not the final production tokenizer. It trains each
requested vocabulary on the same first N non-empty documents and evaluates each on a
fixed sample from the same corpus. Use a larger sample for finalists.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, normalizers

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SYSTEM]", "[USER]", "[ASSISTANT]"]
PROBES = [
    "pneumothorax", "pleural effusion", "atelectasis", "cardiomegaly",
    "pulmonary edema", "consolidation", "ground-glass opacity",
    "interstitial lung disease", "chronic obstructive pulmonary disease",
    "No focal air-space consolidation or pleural effusion.",
]


def iter_texts(data_dir: Path, column: str, limit: int | None):
    files = sorted(data_dir.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No part-*.parquet shards found in {data_dir}")
    seen = 0
    for file in files:
        parquet = pq.ParquetFile(file)
        if column not in parquet.schema_arrow.names:
            raise ValueError(f"Column {column!r} not found in {file}")
        for batch in parquet.iter_batches(columns=[column], batch_size=256):
            for value in batch.column(0).to_pylist():
                if isinstance(value, str) and value.strip():
                    yield value
                    seen += 1
                    if limit is not None and seen >= limit:
                        return


def build_tokenizer(data_dir: Path, column: str, vocab_size: int, min_frequency: int,
                    train_rows: int | None) -> tuple[Tokenizer, float]:
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
    start = time.perf_counter()
    tokenizer.train_from_iterator(
        iter_texts(data_dir, column, train_rows), trainer=trainer, length=None
    )
    return tokenizer, time.perf_counter() - start


def evaluate(tokenizer: Tokenizer, data_dir: Path, column: str, eval_rows: int) -> dict:
    docs = 0
    chars = 0
    token_count = 0
    total_bytes = 0
    for text in iter_texts(data_dir, column, eval_rows):
        encoded = tokenizer.encode(text)
        docs += 1
        chars += len(text)
        total_bytes += len(text.encode("utf-8"))
        token_count += len(encoded.ids)
    return {
        "evaluation_documents": docs,
        "evaluation_characters": chars,
        "evaluation_utf8_bytes": total_bytes,
        "evaluation_tokens": token_count,
        "tokens_per_document": round(token_count / docs, 3) if docs else None,
        "characters_per_token": round(chars / token_count, 4) if token_count else None,
        "bytes_per_token": round(total_bytes / token_count, 4) if token_count else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("/content/drive/MyDrive/LLM/common_pile_pubmed"))
    p.add_argument("--output-dir", type=Path, default=Path("/content/drive/MyDrive/LLM/tokenizer_vocab_benchmark"))
    p.add_argument("--vocab-sizes", default="50000,75000,100000,150000,200000,300000",
                   help="Comma-separated target vocabularies")
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--train-rows", type=int, default=100000,
                   help="Documents per tokenizer; use 10000 for a quick smoke test")
    p.add_argument("--eval-rows", type=int, default=5000)
    p.add_argument("--text-column", default="text")
    args = p.parse_args()

    sizes = [int(x.strip()) for x in args.vocab_sizes.split(",") if x.strip()]
    if not sizes or any(x < 256 for x in sizes):
        p.error("Every --vocab-sizes entry must be at least 256")
    if args.train_rows < 1 or args.eval_rows < 1:
        p.error("--train-rows and --eval-rows must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    print(f"Corpus: {args.data_dir}", flush=True)
    print(f"Targets: {sizes}; train rows/model: {args.train_rows:,}; eval rows: {args.eval_rows:,}", flush=True)
    print("Each tokenizer is trained independently on the same ordered sample.", flush=True)

    for target in sizes:
        out_dir = args.output_dir / f"bpe_{target}"
        out_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_path = out_dir / "tokenizer.json"
        if tokenizer_path.exists():
            print(f"Skipping existing {tokenizer_path}; delete that folder to rerun.", flush=True)
            report_path = out_dir / "benchmark_report.json"
            if report_path.exists():
                results.append(json.loads(report_path.read_text(encoding="utf-8")))
            continue

        print(f"\n=== Training target vocabulary {target:,} ===", flush=True)
        tokenizer, elapsed = build_tokenizer(
            args.data_dir, args.text_column, target, args.min_frequency, args.train_rows
        )
        tokenizer.save(str(tokenizer_path))
        actual = tokenizer.get_vocab_size()
        print(f"Training finished in {elapsed / 60:.2f} min; actual vocab={actual:,}", flush=True)
        metrics = evaluate(tokenizer, args.data_dir, args.text_column, args.eval_rows)
        probe_metrics = {
            term: {
                "token_count": len(tokenizer.encode(term).ids),
                "tokens": tokenizer.encode(term).tokens,
            }
            for term in PROBES
        }
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "algorithm": "BPE",
            "normalizer": "NFKC",
            "pre_tokenizer": "ByteLevel",
            "target_vocab_size": target,
            "actual_vocab_size": actual,
            "min_frequency": args.min_frequency,
            "training_rows_limit": args.train_rows,
            "training_seconds": round(elapsed, 2),
            **metrics,
            "probe_tokenization": probe_metrics,
            "tokenizer_path": str(tokenizer_path),
            "note": "General PubMed sample, not chest-filtered. Use the same held-out evaluation set for a final comparison.",
        }
        (out_dir / "benchmark_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        results.append(report)
        print(
            f"Result: vocab={actual:,} | tokens/doc={metrics['tokens_per_document']} | "
            f"chars/token={metrics['characters_per_token']} | seconds={elapsed:.1f}",
            flush=True,
        )

    summary_path = args.output_dir / "benchmark_summary.csv"
    if results:
        fields = [
            "target_vocab_size", "actual_vocab_size", "training_seconds",
            "evaluation_documents", "evaluation_tokens", "tokens_per_document",
            "characters_per_token", "bytes_per_token",
        ]
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
    print(f"\nSaved summary: {summary_path}", flush=True)
    print("Benchmark complete. Review token savings against vocabulary/embedding cost before selecting a size.", flush=True)


if __name__ == "__main__":
    main()
