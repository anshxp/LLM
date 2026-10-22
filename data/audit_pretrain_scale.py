"""Count usable Seq2Seq denoising tokens with periodic progress logs."""
import argparse
import time
from pathlib import Path

import pyarrow.parquet as pq

from data.parquet_dataset import row_to_text, _split_for_row
from data.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=Path("data/raw/stage 1/train.parquet"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/processed/tokenizer.json"))
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--target-tokens-per-parameter", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--log-every-batches", type=int, default=10)
    args = parser.parse_args()

    if not args.parquet.exists():
        raise FileNotFoundError(args.parquet)
    if args.context_length < 8:
        raise ValueError("context-length must be >= 8")
    if args.target_tokens_per_parameter <= 0:
        raise ValueError("target-tokens-per-parameter must be positive")
    if args.batch_size < 1 or args.log_every_batches < 1:
        raise ValueError("batch-size and log-every-batches must be positive")

    print(f"[START] Opening corpus: {args.parquet}", flush=True)
    print(f"[START] Loading tokenizer: {args.tokenizer}", flush=True)
    load_start = time.perf_counter()
    tokenizer = Tokenizer.from_file(args.tokenizer)
    print(
        f"[READY] Vocabulary={len(tokenizer):,}; "
        f"tokenizer_load_time={time.perf_counter() - load_start:.1f}s",
        flush=True,
    )

    parquet = pq.ParquetFile(args.parquet)
    total_rows = parquet.metadata.num_rows
    total_batches = (total_rows + args.batch_size - 1) // args.batch_size
    counts = {"train": 0, "validation": 0, "test": 0}
    rows = {"train": 0, "validation": 0, "test": 0}
    row_index = 0
    processed_rows = 0
    audit_start = time.perf_counter()
    last_log = audit_start

    print(
        f"[START] Rows={total_rows:,}; batches={total_batches:,}; "
        f"batch_size={args.batch_size}; context={args.context_length}",
        flush=True,
    )

    for batch_num, batch in enumerate(parquet.iter_batches(batch_size=args.batch_size), start=1):
        for row in batch.to_pylist():
            split = _split_for_row(row_index, row)
            row_index += 1
            processed_rows += 1
            text = row_to_text(row)
            if not text:
                continue

            token_ids = tokenizer.encode(text, add_eos=True)
            if len(token_ids) < 8:
                continue

            # Match the pretraining dataset's per-row context truncation.
            counts[split] += min(len(token_ids), args.context_length)
            rows[split] += 1

        if batch_num % args.log_every_batches == 0 or batch_num == total_batches:
            now = time.perf_counter()
            elapsed = now - audit_start
            interval = now - last_log
            rate = processed_rows / elapsed if elapsed > 0 else 0.0
            remaining = max(total_rows - processed_rows, 0)
            eta_seconds = remaining / rate if rate > 0 else 0.0
            pct = 100.0 * processed_rows / total_rows if total_rows else 100.0
            print(
                f"[PROGRESS] {processed_rows:,}/{total_rows:,} rows ({pct:.2f}%) "
                f"| batch {batch_num:,}/{total_batches:,} "
                f"| train_tokens={counts['train']:,} "
                f"| validation_rows={rows['validation']:,} "
                f"| test_rows={rows['test']:,} "
                f"| speed={rate:,.0f} rows/s "
                f"| elapsed={elapsed / 60:.1f} min "
                f"| ETA~{eta_seconds / 60:.1f} min "
                f"| last_log_interval={interval:.1f}s",
                flush=True,
            )
            last_log = now

    elapsed = time.perf_counter() - audit_start
    train_tokens = counts["train"]
    print("[COMPLETE] Audit finished.", flush=True)
    print(
        f"file_mb={args.parquet.stat().st_size / (1024 * 1024):.2f} "
        f"vocab={len(tokenizer):,} context={args.context_length}",
        flush=True,
    )
    for split in ("train", "validation", "test"):
        print(
            f"{split}: usable_rows={rows[split]:,} "
            f"usable_target_tokens_per_pass={counts[split]:,}",
            flush=True,
        )
    print(
        f"reference_parameter_budget={train_tokens / args.target_tokens_per_parameter:,.0f} "
        f"at {args.target_tokens_per_parameter:g} tokens/parameter (heuristic)",
        flush=True,
    )
    print(
        f"total_rows_processed={processed_rows:,}; elapsed_minutes={elapsed / 60:.2f}",
        flush=True,
    )
    print(
        "This budget describes one dataset pass, not repeated-epoch tokens. "
        "Use validation loss and the actual architecture parameter count when choosing model size.",
        flush=True,
    )


if __name__ == "__main__":
    main()
