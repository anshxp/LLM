"""Train a BPE tokenizer directly from the Common Pile PubMed Hugging Face dataset.

Uses Hugging Face streaming so the complete corpus is not downloaded to disk.
The training run itself is not resumable until tokenizers has finished and saved
tokenizer.json; use a small --max-rows smoke test before a full run.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, normalizers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="common-pile/pubmed")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Optional smoke-test cap; omit to consume the full split")
    parser.add_argument("--log-every", type=int, default=10000)
    args = parser.parse_args()

    if args.vocab_size < 256:
        parser.error("--vocab-size must be at least 256")
    if args.min_frequency < 1 or args.log_every < 1:
        parser.error("--min-frequency and --log-every must be positive")
    if args.max_rows is not None and args.max_rows < 1:
        parser.error("--max-rows must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / "tokenizer.json"
    report_file = args.output_dir / "tokenizer_report.json"
    if output_file.exists() or report_file.exists():
        raise FileExistsError(
            f"Output already exists in {args.output_dir}; choose a fresh output directory."
        )

    print(f"[startup] Dataset: {args.dataset} | split={args.split}", flush=True)
    print("[startup] Opening dataset in streaming mode; full corpus will not be downloaded.", flush=True)
    dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    print(f"[startup] Streaming dataset opened. Text column: {args.text_column}", flush=True)

    stats = {"documents": 0, "characters": 0}
    started = time.monotonic()
    last_log = started

    def texts():
        nonlocal last_log
        for row in dataset:
            value = row.get(args.text_column)
            if not isinstance(value, str) or not value.strip():
                continue
            stats["documents"] += 1
            stats["characters"] += len(value)
            now = time.monotonic()
            if stats["documents"] == 1 or stats["documents"] % args.log_every == 0 or now - last_log >= 30:
                elapsed = now - started
                rate = stats["documents"] / elapsed if elapsed else 0.0
                print(
                    f"[input] docs={stats['documents']:,} | chars={stats['characters']:,} "
                    f"| elapsed={elapsed / 60:.1f} min | rate={rate:.1f} docs/s",
                    flush=True,
                )
                last_log = now
            yield value
            if args.max_rows is not None and stats["documents"] >= args.max_rows:
                print(f"[input] Reached --max-rows={args.max_rows:,}; ending smoke test.", flush=True)
                return

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

    print(
        f"[training] Starting BPE training | target_vocab={args.vocab_size:,} "
        f"| max_rows={args.max_rows if args.max_rows is not None else 'ALL'}",
        flush=True,
    )
    training_started = time.monotonic()
    tokenizer.train_from_iterator(texts(), trainer=trainer)
    training_seconds = time.monotonic() - training_started

    tokenizer.save(str(output_file))
    probes = [
        "pneumothorax", "pleural effusion", "atelectasis", "cardiomegaly",
        "pulmonary edema", "consolidation", "ground-glass opacity",
        "No focal air-space consolidation or pleural effusion.",
    ]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "split": args.split,
        "streaming": True,
        "algorithm": "BPE",
        "normalizer": "NFKC",
        "pre_tokenizer": "ByteLevel",
        "target_vocab_size": args.vocab_size,
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": special_tokens,
        "text_column": args.text_column,
        "documents_consumed": stats["documents"],
        "characters_consumed": stats["characters"],
        "max_rows_limit": args.max_rows,
        "training_seconds": round(training_seconds, 3),
        "probe_tokenization": {
            text: {"tokens": tokenizer.encode(text).tokens, "ids": tokenizer.encode(text).ids}
            for text in probes
        },
        "note": "General PubMed corpus; clinical terms are probes, not a corpus filter.",
    }
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] Saved tokenizer: {output_file}", flush=True)
    print(f"[done] Saved report: {report_file}", flush=True)
    print(f"[done] Actual vocabulary size: {tokenizer.get_vocab_size():,}", flush=True)
    print(f"[done] Documents consumed: {stats['documents']:,}", flush=True)
    print(f"[done] Training elapsed: {training_seconds / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
