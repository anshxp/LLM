"""Build the compact corpus artifacts used by v2 training.

Books are cleaned into train/validation/test text splits. The large supervised
corpus is passed through the existing disk-backed Fine tuning 2 pipeline so
JSON/JSONL/CSV/TSV/XML/XLSX/Parquet/ZIP sources can be processed without
loading the complete corpus into RAM.
"""

import argparse
import hashlib
import json
from pathlib import Path

from data.cleaner import clean_text
from data.dedup import is_near_duplicate, simhash
from data.filters import passes_basic_filters
from data.pdf_extractor import extract_pdf_text
from data.finetuning2_pipeline import build as build_finetuning2

SUPPORTED_TEXT = {".txt", ".md", ".markdown"}
SUPPORTED = SUPPORTED_TEXT | {".pdf"}


def _extract(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return extract_pdf_text(path)
    return path.read_text(encoding="utf-8", errors="replace")


def _split_for_hash(value: str) -> str:
    bucket = int(value[:8], 16) % 100
    return "train" if bucket < 90 else "validation" if bucket < 95 else "test"


def build_books(source_root: Path, output_dir: Path) -> dict:
    """Clean, exact-deduplicate and near-deduplicate book/source files."""
    source_root = Path(source_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        p for p in source_root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )
    if not paths:
        raise FileNotFoundError(f"No supported book files found under {source_root}")

    handles = {
        split: (output_dir / f"{split}.txt").open("w", encoding="utf-8")
        for split in ("train", "validation", "test")
    }
    manifest_path = output_dir / "manifest.jsonl"
    seen_hashes = set()
    seen_simhashes = []
    stats = {
        "files": 0,
        "accepted": 0,
        "rejected": 0,
        "duplicates": 0,
        "near_duplicates": 0,
    }

    try:
        with manifest_path.open("w", encoding="utf-8") as manifest:
            for path in paths:
                stats["files"] += 1
                try:
                    text = clean_text(_extract(path))
                except Exception as exc:
                    stats["rejected"] += 1
                    print(f"[books] extraction failed: {path}: {exc}")
                    continue
                if not passes_basic_filters(text):
                    stats["rejected"] += 1
                    continue

                normalized = " ".join(text.split())
                content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                if content_hash in seen_hashes:
                    stats["duplicates"] += 1
                    continue
                seen_hashes.add(content_hash)

                fingerprint = simhash(text)
                if is_near_duplicate(fingerprint, seen_simhashes):
                    stats["near_duplicates"] += 1
                    continue
                seen_simhashes.append(fingerprint)

                split_name = _split_for_hash(content_hash)
                handles[split_name].write(text.strip() + "\n\n")
                manifest.write(
                    json.dumps(
                        {
                            "source": str(path),
                            "content_sha256": content_hash,
                            "simhash": f"{fingerprint:016x}",
                            "characters": len(text),
                            "split": split_name,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                stats["accepted"] += 1
    finally:
        for handle in handles.values():
            handle.close()

    if stats["accepted"] == 0:
        raise RuntimeError("No book documents survived cleaning and deduplication")
    return stats


def build_sft(source_root: Path, output_dir: Path, tokenizer: Path, context_length: int) -> dict:
    """Prepare the large supervised corpus with disk-backed global deduplication."""
    return build_finetuning2(
        Path(source_root),
        Path(output_dir),
        Path(tokenizer),
        context_length,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v2 book-pretraining and SFT corpora.")
    parser.add_argument("--books-root", type=Path, required=True)
    parser.add_argument("--sft-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/v2"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/processed/tokenizer.json"))
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--sft-shard-size", type=int, default=50_000)
    args = parser.parse_args()

    books = build_books(args.books_root, args.output_root / "books")
    sft = build_sft(
        args.sft_root,
        args.output_root / "sft",
        args.tokenizer,
        args.context_length,
    )
    manifest = {"books": books, "sft": sft}
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
