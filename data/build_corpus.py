from pathlib import Path

from data.cleaner import clean_text
from data.file_hash import calculate_sha256
from data.filters import passes_basic_filters
from data.pdf_extractor import extract_pdf_text


RAW_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/processed/corpus.txt")


def build_corpus() -> dict:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()

    total = 0
    duplicates = 0
    extraction_failures = 0
    rejected = 0
    accepted = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        for pdf_path in sorted(RAW_DIR.glob("*.pdf")):
            total += 1

            # Exact duplicate detection
            file_hash = calculate_sha256(pdf_path)

            if file_hash in seen_hashes:
                duplicates += 1
                continue

            seen_hashes.add(file_hash)

            # Extract text
            try:
                text = extract_pdf_text(pdf_path)
            except Exception as exc:
                extraction_failures += 1
                print(f"Extraction failed: {pdf_path.name}: {exc}")
                continue

            # Skip PDFs with essentially no usable text
            if not text.strip():
                rejected += 1
                print(f"Rejected (no text): {pdf_path.name}")
                continue

            # Clean
            cleaned = clean_text(text)

            # Quality filter
            if not passes_basic_filters(cleaned):
                rejected += 1
                print(f"Rejected (quality filter): {pdf_path.name}")
                continue

            # Keep documents separated by two newlines
            output.write(cleaned)
            output.write("\n\n")

            accepted += 1

            print(
                f"Accepted: {pdf_path.name} "
                f"({len(cleaned):,} characters)"
            )

    return {
        "total_pdfs": total,
        "duplicates": duplicates,
        "extraction_failures": extraction_failures,
        "rejected": rejected,
        "accepted": accepted,
    }


if __name__ == "__main__":
    stats = build_corpus()

    print("\nCorpus build complete")
    print("--------------------")

    for key, value in stats.items():
        print(f"{key}: {value}")

    print(f"\nOutput: {OUTPUT_FILE}")