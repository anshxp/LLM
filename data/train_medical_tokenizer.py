"""Train one or more BPE tokenizers from the prepared medical Parquet shards.

The source corpus must already exist as Parquet files with a text column (and optionally
source metadata). Tokenizer JSON outputs should be kept with checkpoints and never replaced
after model training begins.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def parquet_paths(root: Path) -> list[Path]:
    if root.is_file() and root.suffix == ".parquet":
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus path does not exist: {root}")
    paths = sorted(p for p in root.glob("*.parquet")
                   if not p.name.endswith(".tmp") and p.name != "medical_pretrain.parquet")
    merged = root / "medical_pretrain.parquet"
    if merged.exists():
        return [merged]
    if not paths:
        raise FileNotFoundError(f"No Parquet shards found in {root}")
    return paths


def sample_texts(paths: list[Path], max_documents: int, seed: int) -> list[str]:
    """Read a bounded, source-balanced sample without scanning the entire corpus."""
    del seed  # File ordering is deterministic; source-balanced caps make the sample repeatable.
    groups: dict[str, list[Path]] = {}
    for path in paths:
        if path.name == "medical_pretrain.parquet":
            groups.setdefault("merged", []).append(path)
        elif path.name.startswith("medrag_pubmed_"):
            groups.setdefault("medrag_pubmed", []).append(path)
        elif path.name.startswith("scientific_papers_pubmed_"):
            groups.setdefault("scientific_papers_pubmed", []).append(path)
        else:
            groups.setdefault(path.stem.split("_")[0], []).append(path)

    quota = max(1, max_documents // max(1, len(groups)))
    sample: list[str] = []
    for source, source_paths in sorted(groups.items()):
        limit = max_documents if source == "merged" else quota
        collected = 0
        for path in source_paths:
            parquet = pq.ParquetFile(path)
            if "text" not in parquet.schema_arrow.names:
                raise ValueError(f"Parquet file has no 'text' column: {path}")
            for batch in parquet.iter_batches(columns=["text"], batch_size=512):
                for value in batch.column(0).to_pylist():
                    text = str(value or "").strip()
                    if len(text) < 20:
                        continue
                    sample.append(text)
                    collected += 1
                    if collected >= limit or len(sample) >= max_documents:
                        break
                if collected >= limit or len(sample) >= max_documents:
                    break
            if collected >= limit or len(sample) >= max_documents:
                break
        print(f"Tokenizer sample from {source}: {collected:,} documents")
    if not sample:
        raise ValueError("No usable text found in Parquet corpus.")
    return sample

def batch_iterator(texts: list[str], batch_size: int = 256) -> Iterator[list[str]]:
    for i in range(0, len(texts), batch_size):
        yield texts[i:i + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path,
                        default=Path("/content/drive/MyDrive/LLM/medical_corpus_full"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/content/drive/MyDrive/LLM/tokenizers"))
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[10000, 16000, 32000])
    parser.add_argument("--max-documents", type=int, default=100000,
                        help="Maximum documents used to train each tokenizer")
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_documents <= 0 or args.min_frequency <= 0:
        parser.error("--max-documents and --min-frequency must be positive")
    if any(v <= len(SPECIAL_TOKENS) for v in args.vocab_sizes):
        parser.error("Every vocabulary size must exceed the number of special tokens")

    paths = parquet_paths(args.corpus_dir)
    print(f"Found {len(paths):,} Parquet file(s). Sampling up to {args.max_documents:,} documents.")
    texts = sample_texts(paths, args.max_documents, args.seed)
    print(f"Tokenizer training sample: {len(texts):,} documents.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"corpus_path": str(args.corpus_dir), "sample_documents": len(texts),
              "seed": args.seed, "candidates": []}

    for vocab_size in args.vocab_sizes:
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS,
                             min_frequency=args.min_frequency, show_progress=True)
        tokenizer.train_from_iterator(
            (text for batch in batch_iterator(texts) for text in batch),
            trainer=trainer,
            length=len(texts),
        )
        actual = tokenizer.get_vocab_size()
        out = args.output_dir / f"medical_bpe_{actual}.json"
        tokenizer.save(str(out))
        # Measure token efficiency on the same sample for fair comparison.
        token_count = 0
        char_count = 0
        unk_count = 0
        for text in texts:
            encoded = tokenizer.encode(text)
            token_count += len(encoded.ids)
            char_count += len(text)
            unk_id = tokenizer.token_to_id("<unk>")
            unk_count += sum(token_id == unk_id for token_id in encoded.ids)
        item = {"requested_vocab_size": vocab_size, "actual_vocab_size": actual,
                "tokenizer_path": str(out), "sample_tokens": token_count,
                "sample_characters": char_count,
                "characters_per_token": round(char_count / max(1, token_count), 4),
                "unknown_token_count": unk_count}
        report["candidates"].append(item)
        print(json.dumps(item, ensure_ascii=False))

    report_path = args.output_dir / "tokenizer_comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Comparison report: {report_path}")
    print("Choose a tokenizer before model initialization; do not replace it after training starts.")


if __name__ == "__main__":
    main()
