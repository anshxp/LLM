"""Storage-efficient streaming corpus pipeline for v2 pretraining."""
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
from data.pdf_extractor import iter_pdf_paragraphs

PRETRAIN_ROOT = Path("data/raw")
SUPPORTED_TEXT = {".txt", ".text", ".md", ".markdown"}
SUPPORTED = SUPPORTED_TEXT | {".pdf", ".parquet"}


def list_pretraining_shards(source_root: Path = PRETRAIN_ROOT) -> list[Path]:
    paths = sorted(
        p for p in Path(source_root).rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )
    if not paths:
        raise FileNotFoundError(f"No supported pretraining files found under {source_root}")
    return paths


def _as_text(value) -> Iterator[str]:
    """Handle BlueScrubs rows stored as either string or list-of-strings."""
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            if item is not None:
                yield str(item)
    else:
        yield str(value)


def _iter_parquet_text(path: Path) -> Iterator[str]:
    """Stream only the text column in small Arrow batches."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    if "text" not in parquet.schema_arrow.names:
        raise ValueError(f"Parquet pretraining file has no 'text' column: {path}")
    for batch in parquet.iter_batches(batch_size=128, columns=["text"]):
        for value in batch.column(0).to_pylist():
            yield from _as_text(value)


def iter_shard_text(path: Path) -> Iterator[str]:
    """Stream one raw shard at a time with bounded memory."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        source = _iter_parquet_text(path)
    elif suffix == ".pdf":
        for paragraph in iter_pdf_paragraphs(path):
            text = clean_text(paragraph)
            if passes_basic_filters(text):
                yield text
        return
    else:
        text = clean_text(path.read_text(encoding="utf-8", errors="replace"))
        if passes_basic_filters(text):
            yield text
        return

    for text in source:
        text = clean_text(text)
        if passes_basic_filters(text):
            yield text


def iter_pretraining_text(source_root: Path = PRETRAIN_ROOT) -> Iterator[tuple[str, str]]:
    source_root = Path(source_root)
    for path in list_pretraining_shards(source_root):
        for text in iter_shard_text(path):
            yield text, str(path.relative_to(source_root))


def iter_pretraining_shards(source_root: Path = PRETRAIN_ROOT) -> Iterator[tuple[str, Iterator[str]]]:
    source_root = Path(source_root)
    for path in list_pretraining_shards(source_root):
        yield str(path.relative_to(source_root)), iter_shard_text(path)


def build_pretraining_manifest(
    source_root: Path = PRETRAIN_ROOT,
    manifest_path: Path = Path("data/processed/v2/pretrain_manifest.jsonl"),
) -> dict:
    """Create compact metadata; the raw 57GB corpus is never copied."""
    source_root = Path(source_root).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    db_path = manifest_path.parent / ".pretrain_dedup.sqlite3"
    seen_near: list[int] = []
    stats = {
        "files": 0,
        "documents_seen": 0,
        "accepted": 0,
        "rejected": 0,
        "duplicates": 0,
        "near_duplicates": 0,
        "extraction_failures": 0,
    }

    db = sqlite3.connect(db_path)
    db.execute("CREATE TABLE IF NOT EXISTS seen (sha256 BLOB PRIMARY KEY)")
    db.commit()
    try:
        with manifest_path.open("w", encoding="utf-8") as manifest:
            for source, documents in iter_pretraining_shards(source_root):
                stats["files"] += 1
                try:
                    for text in documents:
                        stats["documents_seen"] += 1
                        normalized = " ".join(text.split())
                        digest = hashlib.sha256(normalized.encode("utf-8")).digest()
                        if not db.execute("INSERT OR IGNORE INTO seen VALUES (?)", (digest,)).rowcount:
                            stats["duplicates"] += 1
                            continue
                        if source.lower().endswith((".txt", ".text", ".md", ".markdown", ".pdf")):
                            fingerprint = simhash(text)
                            if is_near_duplicate(fingerprint, seen_near):
                                stats["near_duplicates"] += 1
                                continue
                            seen_near.append(fingerprint)
                        manifest.write(
                            json.dumps(
                                {"source": source, "sha256": digest.hex(), "characters": len(text)},
                                ensure_ascii=False,
                            ) + "\n"
                        )
                        stats["accepted"] += 1
                except Exception as exc:
                    stats["extraction_failures"] += 1
                    print(f"Skipped unreadable shard: {source}: {exc}")
            db.commit()
    finally:
        db.close()
        for suffix in ("", "-wal", "-shm"):
            (Path(str(db_path) + suffix)).unlink(missing_ok=True)

    manifest_path.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def build_sft(*_args, **_kwargs):
    raise RuntimeError("SFT is disabled during pretraining. Start SFT explicitly after pretraining.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact metadata for streaming v2 pretraining.")
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/v2"))
    args = parser.parse_args()
    if not PRETRAIN_ROOT.is_dir():
        raise FileNotFoundError("Pretraining corpus not found at data/raw.")
    stats = build_pretraining_manifest(PRETRAIN_ROOT, args.output_root / "pretrain_manifest.jsonl")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
