"""Storage-efficient corpus pipeline for v2 pretraining.

Repository layout is fixed:
    data/raw/             all pretraining sources
    data/Fine tuning 2/   supervised fine-tuning sources (not built here)

Pretraining streams directly from data/raw. Large Parquet shards are processed
one at a time and no processed copy of the corpus is created.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Iterator

from data.cleaner import clean_text
from data.dedup import is_near_duplicate, simhash
from data.filters import passes_basic_filters
from data.pdf_extractor import extract_pdf_text

PRETRAIN_ROOT = Path("data/raw")
SFT_ROOT = Path("data/Fine tuning 2")
SUPPORTED_TEXT = {".txt", ".md", ".markdown"}
SUPPORTED = SUPPORTED_TEXT | {".pdf", ".parquet"}


def list_pretraining_shards(source_root: Path = PRETRAIN_ROOT) -> list[Path]:
    """Return deterministic pretraining files; each file is a resumable unit."""
    paths = sorted(
        p for p in Path(source_root).rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )
    if not paths:
        raise FileNotFoundError(f"No supported pretraining files found under {source_root}")
    return paths


def _iter_parquet_text(path: Path) -> Iterator[str]:
    """Yield only the text column in small Arrow batches."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    if "text" not in parquet.schema_arrow.names:
        raise ValueError(f"Parquet pretraining file has no 'text' column: {path}")
    for batch in parquet.iter_batches(batch_size=128, columns=["text"]):
        for value in batch.column(0).to_pylist():
            if value is not None:
                yield str(value)


def iter_shard_text(path: Path) -> Iterator[str]:
    """Stream one raw shard at a time with bounded memory."""
    if path.suffix.lower() == ".parquet":
        source = _iter_parquet_text(path)
    elif path.suffix.lower() == ".pdf":
        yield clean_text(extract_pdf_text(path))
        return
    else:
        yield clean_text(path.read_text(encoding="utf-8", errors="replace"))
        return

    for text in source:
        text = clean_text(text)
        if passes_basic_filters(text):
            yield text


def iter_pretraining_text(source_root: Path = PRETRAIN_ROOT) -> Iterator[tuple[str, str]]:
    """Stream all pretraining documents without creating a processed corpus."""
    source_root = Path(source_root)
    for path in list_pretraining_shards(source_root):
        for text in iter_shard_text(path):
            if passes_basic_filters(text):
                yield text, str(path.relative_to(source_root))


def iter_pretraining_shards(source_root: Path = PRETRAIN_ROOT) -> Iterator[tuple[str, Iterator[str]]]:
    """Yield (relative shard name, document iterator) one shard at a time."""
    source_root = Path(source_root)
    for path in list_pretraining_shards(source_root):
        yield str(path.relative_to(source_root)), iter_shard_text(path)


def build_pretraining_manifest(
    source_root: Path = PRETRAIN_ROOT,
    manifest_path: Path = Path("data/processed/v2/pretrain_manifest.jsonl"),
) -> dict:
    """Create only tiny metadata; raw training text stays in data/raw."""
    source_root = Path(source_root).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    db_path = manifest_path.parent / ".pretrain_dedup.sqlite3"
    seen_near: list[int] = []
    stats = {"files": 0, "documents_seen": 0, "accepted": 0, "rejected": 0, "duplicates": 0, "near_duplicates": 0}

    db = sqlite3.connect(db_path)
    db.execute("CREATE TABLE IF NOT EXISTS seen (sha256 BLOB PRIMARY KEY)")
    db.commit()
    try:
        with manifest_path.open("w", encoding="utf-8") as manifest:
            for source, documents in iter_pretraining_shards(source_root):
                stats["files"] += 1
                for text in documents:
                    stats["documents_seen"] += 1
                    normalized = " ".join(text.split())
                    digest = hashlib.sha256(normalized.encode("utf-8")).digest()
                    if not db.execute("INSERT OR IGNORE INTO seen VALUES (?)", (digest,)).rowcount:
                        stats["duplicates"] += 1
                        continue
                    if source.lower().endswith((".txt", ".md", ".markdown", ".pdf")):
                        fingerprint = simhash(text)
                        if is_near_duplicate(fingerprint, seen_near):
                            stats["near_duplicates"] += 1
                            continue
                        seen_near.append(fingerprint)
                    manifest.write(json.dumps({"source": source, "sha256": digest.hex(), "characters": len(text)}, ensure_ascii=False) + "\n")
                    stats["accepted"] += 1
            db.commit()
    finally:
        db.close()
        for suffix in ("", "-wal", "-shm"):
            (Path(str(db_path) + suffix)).unlink(missing_ok=True)

    manifest_path.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def build_sft(*_args, **_kwargs):
    """SFT is disabled until the user explicitly starts the SFT phase."""
    raise RuntimeError("SFT is disabled during pretraining. Start SFT explicitly after pretraining.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare tiny metadata for streaming v2 pretraining.")
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/v2"))
    args = parser.parse_args()
    if not PRETRAIN_ROOT.is_dir():
        raise FileNotFoundError("Pretraining corpus not found at data/raw.")
    stats = build_pretraining_manifest(PRETRAIN_ROOT, args.output_root / "pretrain_manifest.jsonl")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
