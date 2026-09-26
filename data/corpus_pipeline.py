"""Storage-efficient corpus pipeline for v2 pretraining.

Repository layout is fixed:
    data/raw/             all pretraining sources
    data/Fine tuning 2/   supervised fine-tuning sources (not built here)

Pretraining is intentionally streamed directly from data/raw. No processed
copy of the large corpus is created. This keeps disk usage and RAM bounded.
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


def _iter_parquet_text(path: Path) -> Iterator[str]:
    """Yield Parquet text rows in small Arrow batches."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    if "text" not in parquet.schema_arrow.names:
        raise ValueError(f"Parquet pretraining file has no 'text' column: {path}")
    for batch in parquet.iter_batches(batch_size=256, columns=["text"]):
        for value in batch.column(0).to_pylist():
            if value is not None:
                yield str(value)


def iter_pretraining_text(source_root: Path = PRETRAIN_ROOT) -> Iterator[tuple[str, str]]:
    """Stream cleaned pretraining documents without creating a processed corpus.

    The caller receives (text, source) one document at a time. Large Parquet
    shards are never loaded into RAM and no duplicate processed text files are
    written to disk.
    """
    source_root = Path(source_root)
    paths = sorted(
        p for p in source_root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )
    if not paths:
        raise FileNotFoundError(f"No supported pretraining files found under {source_root}")

    for path in paths:
        if path.suffix.lower() == ".parquet":
            for text in _iter_parquet_text(path):
                text = clean_text(text)
                if passes_basic_filters(text):
                    yield text, str(path.relative_to(source_root))
        else:
            if path.suffix.lower() == ".pdf":
                text = extract_pdf_text(path)
            else:
                text = path.read_text(encoding="utf-8", errors="replace")
            text = clean_text(text)
            if passes_basic_filters(text):
                yield text, str(path.relative_to(source_root))


def build_pretraining_manifest(
    source_root: Path = PRETRAIN_ROOT,
    manifest_path: Path = Path("data/processed/v2/pretrain_manifest.jsonl"),
) -> dict:
    """Create only a tiny metadata manifest; the training text stays in data/raw.

    Exact duplicate detection is disk-backed and temporary. Near-duplicate
    detection is applied only to ordinary book-sized files, not millions of
    Parquet rows, to keep RAM bounded.
    """
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
            seen_files = set()
            for text, source in iter_pretraining_text(source_root):
                stats["documents_seen"] += 1
                if source not in seen_files:
                    seen_files.add(source)
                    stats["files"] += 1
                normalized = " ".join(text.split())
                digest = hashlib.sha256(normalized.encode("utf-8")).digest()
                if not db.execute("INSERT OR IGNORE INTO seen VALUES (?)", (digest,)).rowcount:
                    stats["duplicates"] += 1
                    continue

                # Only apply the in-memory near-duplicate check to non-Parquet
                # documents. The large medical corpus remains memory bounded.
                if source.lower().endswith((".txt", ".md", ".markdown", ".pdf")):
                    fingerprint = simhash(text)
                    if is_near_duplicate(fingerprint, seen_near):
                        stats["near_duplicates"] += 1
                        continue
                    seen_near.append(fingerprint)
                else:
                    fingerprint = None

                manifest.write(json.dumps({
                    "source": source,
                    "sha256": digest.hex(),
                    "characters": len(text),
                }, ensure_ascii=False) + "\n")
                stats["accepted"] += 1
            db.commit()
    finally:
        db.close()
        for suffix in ("", "-wal", "-shm"):
            (Path(str(db_path) + suffix)).unlink(missing_ok=True)

    manifest_path.with_suffix(".json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    return stats


def build_sft(*_args, **_kwargs):
    """SFT is deliberately not built during the pretraining phase.

    SFT preparation will be enabled when the user explicitly starts the SFT
    phase. This prevents the large supervised corpus from consuming storage or
    RAM during the current pretraining run.
    """
    raise RuntimeError("SFT is disabled during pretraining. Run it only when the SFT phase is explicitly started.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare metadata for streaming v2 pretraining.")
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/v2"))
    args = parser.parse_args()

    if not PRETRAIN_ROOT.is_dir():
        raise FileNotFoundError("Pretraining corpus not found at data/raw.")

    stats = build_pretraining_manifest(
        PRETRAIN_ROOT, args.output_root / "pretrain_manifest.jsonl"
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
