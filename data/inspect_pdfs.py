from pathlib import Path

from data.pdf_extractor import inspect_pdf


def inspect_pdf_directory(data_dir: str | Path) -> list[dict]:
    """Inspect every PDF in a directory."""
    data_dir = Path(data_dir)

    results = []

    for pdf_path in sorted(data_dir.rglob("*.pdf")):
        try:
            info = inspect_pdf(pdf_path)
            results.append(info)
        except Exception as exc:
            results.append(
                {
                    "file": str(pdf_path),
                    "pages": None,
                    "characters": None,
                    "words": None,
                    "has_text": False,
                    "error": str(exc),
                }
            )

    return results


def print_report(results: list[dict]) -> None:
    """Print a compact PDF inspection report."""
    print(f"PDFs inspected: {len(results)}")
    print()

    print(
        f"{'File':<55}"
        f"{'Pages':>8}"
        f"{'Words':>12}"
        f"{'Text':>8}"
    )

    print("-" * 83)

    for result in results:
        filename = Path(result["file"]).name

        if len(filename) > 52:
            filename = filename[:49] + "..."

        pages = result["pages"] if result["pages"] is not None else "-"
        words = result["words"] if result["words"] is not None else "-"
        has_text = "YES" if result["has_text"] else "NO"

        print(
            f"{filename:<55}"
            f"{str(pages):>8}"
            f"{str(words):>12}"
            f"{has_text:>8}"
        )


if __name__ == "__main__":
    results = inspect_pdf_directory("data/raw")
    print_report(results)