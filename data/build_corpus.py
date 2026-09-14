import hashlib
import json
from pathlib import Path

from data.cleaner import clean_text
from data.file_hash import calculate_sha256
from data.filters import passes_basic_filters
from data.ingestion import SUPPORTED_TEXT_EXTENSIONS, iter_supported_files
from data.pdf_extractor import extract_pdf_text


RAW_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/processed/corpus.txt")
MANIFEST_FILE = Path("data/processed/manifest.jsonl")


def _extract(path: Path) -> str:
    """Extract raw text from a supported source file."""
    if path.suffix.lower() == ".pdf":
        return extract_pdf_text(path)
    if path.suffix.lower() in SUPPORTED_TEXT_EXTENSIONS:
        return path.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _content_hash(text: str) -> str:
    """Hash normalized text so duplicate documents share one identity."""
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_corpus() -> dict:
    """Build a cleaned corpus and provenance manifest from raw sources."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    seen_file_hashes: set[str] = set()
    seen_content_hashes: set[str] = set()

    stats = {
        "total_files": 0,
        "exact_duplicates": 0,
        "content_duplicates": 0,
        "extraction_failures": 0,
        "rejected": 0,
        "accepted": 0,
    }

    with (
        OUTPUT_FILE.open("w", encoding="utf-8") as output,
        MANIFEST_FILE.open("w", encoding="utf-8") as manifest,
    ):
        for source_path in iter_supported_files(RAW_DIR):
            stats["total_files"] += 1
            file_hash = calculate_sha256(source_path)

            if file_hash in seen_file_hashes:
                stats["exact_duplicates"] += 1
                continue
            seen_file_hashes.add(file_hash)

            try:
                text = _extract(source_path)
            except Exception as exc:
                stats["extraction_failures"] += 1
                print(f"Extraction failed: {source_path}: {exc}")
                continue

            cleaned = clean_text(text)
            if not passes_basic_filters(cleaned):
                stats["rejected"] += 1
                continue

            content_hash = _content_hash(cleaned)
            if content_hash in seen_content_hashes:
                stats["content_duplicates"] += 1
                continue
            seen_content_hashes.add(content_hash)

            output.write(cleaned)
            output.write("\n\n")
            manifest.write(
                json.dumps(
                    {
                        "source": str(source_path),
                        "file_sha256": file_hash,
                        "content_sha256": content_hash,
                        "characters": len(cleaned),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            stats["accepted"] += 1

    return stats


if __name__ == "__main__":
    stats = build_corpus()
    print("\nCorpus build complete")
    print("--------------------")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"Manifest: {MANIFEST_FILE}")
