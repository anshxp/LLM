"""Build the compact v2 corpus artifacts.

Repository data layout is intentionally fixed:

    data/raw/            all pretraining sources (books + parquet corpora)
    data/Fine tuning 2/  supervised fine-tuning sources

The large source corpora are never committed to Git. This module only creates
processed training artifacts under data/processed/v2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from data.cleaner import clean_text
from data.dedup import is_near_duplicate, simhash
from data.filters import passes_basic_filters
from data.pdf_extractor import extract_pdf_text
from data.finetuning2_pipeline import build as build_finetuning2

# Fixed repository layout. Keep these paths stable for Colab and local training.
PRETRAIN_ROOT = Path("data/raw")
SFT_ROOT = Path("data/Fine tuning 2")

SUPPORTED_TEXT = {".txt", ".md", ".markdown"}
SUPPORTED = SUPPORTED_TEXT | {".pdf", ".parquet"}


def _extract(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return extract_pdf_text(path)
    return path.read_text(encoding="utf-8", errors="replace")


def _split_for_hash(value: str) -> str:
    bucket = int(value[:8], 16) % 100
    return "train" if bucket < 90 else "validation" if bucket < 95 else "test"


def _iter_parquet_text(path: Path):
    """Stream only the text column from a local Parquet pretraining shard."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    if "text" not in parquet.schema_arrow.names:
        raise ValueError(f"Parquet pretraining file has no 'text' column: {path}")
    for batch in parquet.iter_batches(batch_size=2048, columns=["text"]):
        for value in batch.column(0).to_pylist():
            if value is not None:
                yield str(value)


def build_books(
    source_root: Path = PRETRAIN_ROOT,
    output_dir: Path = Path("data/processed/v2/books"),
) -> dict:
    """Clean and deduplicate all pretraining material found under data/raw.

    Parquet is streamed in batches so the 57 GB TheBlueScrubs corpus does not
    get loaded into RAM. Its documented train field is ``text``. Exact
    deduplication is disk-backed for every source; in-memory near-duplicate
    detection is used only for non-Parquet book files so it cannot grow with
    the multi-billion-token corpus.
    """
    source_root = Path(source_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        p for p in source_root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )
    if not paths:
        raise FileNotFoundError(f"No supported pretraining files found under {source_root}")

    handles = {
        split: (output_dir / f"{split}.txt").open("w", encoding="utf-8")
        for split in ("train", "validation", "test")
    }
    manifest_path = output_dir / "manifest.jsonl"
    # Temporary disk-backed index. It is deleted after a successful or failed run
    # so the processed corpus does not retain another huge copy of the dataset.
    dedup_db = output_dir / ".dedup.sqlite3"
    seen_near = []
    stats = {
        "files": 0,
        "accepted": 0,
        "rejected": 0,
        "duplicates": 0,
        "near_duplicates": 0,
        "parquet_rows_seen": 0,
    }

    connection = sqlite3.connect(dedup_db)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE IF NOT EXISTS seen (content_sha256 BLOB PRIMARY KEY)")
    connection.commit()

    def process_text(text: str, path: Path, manifest) -> None:
        text = clean_text(text)
        if not passes_basic_filters(text):
            stats["rejected"] += 1
            return

        normalized = " ".join(text.split())
        content_hash = hashlib.sha256(normalized.encode("utf-8")).digest()
        inserted = connection.execute(
            "INSERT OR IGNORE INTO seen(content_sha256) VALUES (?)", (content_hash,)
        ).rowcount
        if not inserted:
            stats["duplicates"] += 1
            return

        # Avoid retaining millions of SimHash values for the large Parquet corpus.
        if path.suffix.lower() != ".parquet":
            fingerprint = simhash(text)
            if is_near_duplicate(fingerprint, seen_near):
                stats["near_duplicates"] += 1
                return
            seen_near.append(fingerprint)
        else:
            fingerprint = None

        split_name = _split_for_hash(content_hash.hex())
        handles[split_name].write(text.strip() + "\n\n")
        entry = {
            "source": str(path.relative_to(source_root)),
            "content_sha256": content_hash.hex(),
            "characters": len(text),
            "split": split_name,
        }
        if fingerprint is not None:
            entry["simhash"] = f"{fingerprint:016x}"
        manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")
        stats["accepted"] += 1

    try:
        with manifest_path.open("w", encoding="utf-8") as manifest:
            for path in paths:
                stats["files"] += 1
                try:
                    if path.suffix.lower() == ".parquet":
                        for text in _iter_parquet_text(path):
                            stats["parquet_rows_seen"] += 1
                            process_text(text, path, manifest)
                    else:
                        process_text(_extract(path), path, manifest)
                    connection.commit()
                except Exception as exc:
                    stats["rejected"] += 1
                    print(f"[pretrain] extraction failed: {path}: {exc}")
    finally:
        connection.commit()
        connection.close()
        for handle in handles.values():
            handle.close()
        for suffix in ("", "-wal", "-shm"):
            (output_dir / f".dedup.sqlite3{suffix}").unlink(missing_ok=True)

    if stats["accepted"] == 0:
        raise RuntimeError("No pretraining documents survived cleaning and deduplication")
    return stats


def build_sft(
    source_root: Path = SFT_ROOT,
    output_dir: Path = Path("data/processed/v2/sft"),
    tokenizer: Path = Path("data/processed/tokenizer.json"),
    context_length: int = 256,
    shard_size: int = 50_000,
) -> dict:
    """Prepare the supervised corpus from the fixed Fine tuning 2 directory."""
    return build_finetuning2(
        Path(source_root), Path(output_dir), Path(tokenizer), context_length, shard_size
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v2 pretraining and SFT corpora.")
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/v2"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/processed/tokenizer.json"))
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--sft-shard-size", type=int, default=50_000)
    args = parser.parse_args()

    if not PRETRAIN_ROOT.is_dir():
        raise FileNotFoundError(
            "Pretraining corpus not found at data/raw. Put all book and pretraining "
            "files under data/raw before running the pipeline."
        )
    if not SFT_ROOT.is_dir():
        raise FileNotFoundError(
            "Fine tuning 2 corpus not found at data/Fine tuning 2."
        )

    books = build_books(PRETRAIN_ROOT, args.output_root / "books")
    sft = build_sft(
        SFT_ROOT,
        args.output_root / "sft",
        args.tokenizer,
        args.context_length,
        args.sft_shard_size,
    )
    manifest = {
        "pretraining_root": str(PRETRAIN_ROOT),
        "sft_root": str(SFT_ROOT),
        "books": books,
        "sft": sft,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
