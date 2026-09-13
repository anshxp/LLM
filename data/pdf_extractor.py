from pathlib import Path
import pymupdf


def extract_pdf_text(pdf_path: str | Path) -> str:
    """
    Extract text from a PDF page by page.

    Returns the combined extracted text.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF does not exist: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {pdf_path}")

    pages = []

    with pymupdf.open(pdf_path) as document:
        for page in document:
            text = page.get_text("text")
            pages.append(text)

    return "\n".join(pages)


def inspect_pdf(pdf_path: str | Path) -> dict:
    """
    Inspect basic PDF extraction characteristics without
    writing any files.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF does not exist: {pdf_path}")

    with pymupdf.open(pdf_path) as document:
        page_count = len(document)

        extracted_text = "\n".join(
            page.get_text("text")
            for page in document
        )

    character_count = len(extracted_text)
    word_count = len(extracted_text.split())

    return {
        "file": str(pdf_path),
        "pages": page_count,
        "characters": character_count,
        "words": word_count,
        "has_text": bool(extracted_text.strip()),
    }


def inspect_pdf_images(pdf_path: str | Path) -> dict:
    """
    Inspect how many images are present in each PDF page.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF does not exist: {pdf_path}")

    with pymupdf.open(pdf_path) as document:
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

if __name__ == "__main__":
    pdf_path = "data/raw/040515.pdf"

    info = inspect_pdf(pdf_path)

    print(f"File: {info['file']}")
    print(f"Pages: {info['pages']}")
    print(f"Characters: {info['characters']}")
    print(f"Words: {info['words']}")
    print(f"Has extractable text: {info['has_text']}")