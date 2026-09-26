from pathlib import Path
from typing import Iterator

import pymupdf


def _open_pdf(pdf_path: Path):
    """Open a PDF reliably from bytes and give a useful error for bad files."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF does not exist: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {pdf_path}")

    raw = pdf_path.read_bytes()
    if not raw.startswith(b"%PDF-"):
        raise ValueError(
            f"Invalid PDF file (missing %PDF header): {pdf_path}"
        )

    try:
        return pymupdf.open(stream=raw, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Unreadable/corrupt PDF: {pdf_path}: {exc}") from exc


def iter_pdf_paragraphs(pdf_path: str | Path) -> Iterator[str]:
    """Yield paragraph-like text blocks without loading the whole PDF into RAM."""
    pdf_path = Path(pdf_path)
    with _open_pdf(pdf_path) as document:
        for page in document:
            blocks = page.get_text("blocks", sort=True)
            for block in blocks:
                text = block[4].strip()
                if text:
                    yield text


def extract_pdf_text(pdf_path: str | Path) -> str:
    """Extract PDF text while preserving paragraph boundaries."""
    return "\n\n".join(iter_pdf_paragraphs(pdf_path))


def inspect_pdf(pdf_path: str | Path) -> dict:
    """Inspect basic PDF extraction characteristics without writing files."""
    pdf_path = Path(pdf_path)
    paragraphs = list(iter_pdf_paragraphs(pdf_path))
    extracted_text = "\n\n".join(paragraphs)
    with _open_pdf(pdf_path) as document:
        page_count = len(document)

    return {
        "file": str(pdf_path),
        "pages": page_count,
        "paragraphs": len(paragraphs),
        "characters": len(extracted_text),
        "words": len(extracted_text.split()),
        "has_text": bool(extracted_text.strip()),
    }


def inspect_pdf_images(pdf_path: str | Path) -> dict:
    """Inspect how many images are present in each PDF page."""
    pdf_path = Path(pdf_path)
    with _open_pdf(pdf_path) as document:
        page_count = len(document)
        image_count = 0
        pages_with_images = 0
        for page in document:
            images = page.get_images(full=True)
            if images:
                pages_with_images += 1
                image_count += len(images)

    return {
        "file": str(pdf_path),
        "pages": page_count,
        "images": image_count,
        "pages_with_images": pages_with_images,
    }
