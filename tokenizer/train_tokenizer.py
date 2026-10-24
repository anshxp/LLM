"""Train a new BPE tokenizer from the Common Pile PubMed Parquet shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, normalizers


def iter_texts(data_dir: Path, text_column: str, max_rows: int | None):
    files = sorted(data_dir.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No part-*.parquet shards found in {data_dir}")
    seen = 0
    for file in files:
        parquet = pq.ParquetFile(file)
        if text_column not in parquet.schema_arrow.names:
            raise ValueError(f"Column {text_column!r} not found in {file}")
        for batch in parquet.iter_batches(columns=[text_column], batch_size=256):
            for value in batch.column(0).to_pylist():
                if isinstance(value, str) and value.strip():
                    yield value
                    seen += 1
                    if max_rows is not None and seen >= max_rows:
                        return


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("/content/drive/MyDrive/LLM/common_pile_pubmed"))
    p.add_argument("--output-dir", type=Path, default=Path("/content/drive/MyDrive/LLM/tokenizer_chest_bpe32k"))
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test limit; omit for all available shards")
    p.add_argument("--text-column", default="text")
    args = p.parse_args()

    if args.vocab_size < 256:
        p.error("--vocab-size must be at least 256")
    if args.min_frequency < 1:
        p.error("--min-frequency must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / "tokenizer.json"
    if output_file.exists():
        raise FileExistsError(f"{output_file} already exists; choose another output directory or move the old tokenizer first.")

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
    print(f"Vocab target: {args.vocab_size:,}; minimum frequency: {args.min_frequency}", flush=True)
    print("Reading Parquet text incrementally; corpus files are not copied into RAM.", flush=True)
    tokenizer.train_from_iterator(
        iter_texts(args.data_dir, args.text_column, args.max_rows),
        trainer=trainer,
        length=None,
    )
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
        "max_rows_limit": args.max_rows,
        "probe_tokenization": {
            text: {"tokens": tokenizer.encode(text).tokens, "ids": tokenizer.encode(text).ids}
            for text in probes
        },
        "note": "Chest terms are evaluation probes; this corpus is general PubMed, not chest-filtered.",
    }
    (args.output_dir / "tokenizer_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved tokenizer: {output_file}", flush=True)
    print(f"Actual vocabulary size: {tokenizer.get_vocab_size():,}", flush=True)
    print(f"Saved evaluation report: {args.output_dir / 'tokenizer_report.json'}", flush=True)


if __name__ == "__main__":
    main()
