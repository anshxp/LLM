from pathlib import Path

from data.pdf_extractor import inspect_pdf, inspect_pdf_images


if __name__ == "__main__":
    data_dir = Path("data/raw")

    for pdf_path in sorted(data_dir.glob("*.pdf")):
        text_info = inspect_pdf(pdf_path)

        if text_info["words"] == 0:
            image_info = inspect_pdf_images(pdf_path)

            print(
                f"{pdf_path.name}: "
                f"{image_info['pages']} pages, "
                f"{image_info['images']} images, "
                f"{image_info['pages_with_images']} pages with images"
            )