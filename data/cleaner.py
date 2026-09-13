import re
import unicodedata


def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters while preserving text content."""
    return unicodedata.normalize("NFKC", text)


def remove_control_characters(text: str) -> str:
    """Remove control characters while preserving newline and tab."""
    return "".join(
        char
        for char in text
        if char in ("\n", "\t") or not unicodedata.category(char).startswith("C")
    )


def normalize_line_endings(text: str) -> str:
    """Normalize Windows and old-style line endings to '\\n'."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_whitespace(text: str) -> str:
    """
    Normalize horizontal whitespace while preserving paragraph structure.
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_text(text: str) -> str:
    """
    Apply the standard text-cleaning pipeline.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = normalize_unicode(text)
    text = normalize_line_endings(text)
    text = remove_control_characters(text)
    text = normalize_whitespace(text)

    return text