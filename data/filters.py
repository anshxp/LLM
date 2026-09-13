import re


def is_nonempty(text: str) -> bool:
    """Return True if the document contains non-whitespace text."""
    return bool(text.strip())


def meets_minimum_length(text: str, min_characters: int = 200) -> bool:
    """Return True if the document has enough textual content."""
    return len(text.strip()) >= min_characters


def has_reasonable_text_ratio(
    text: str,
    min_ratio: float = 0.70,
) -> bool:
    """
    Check whether a reasonable proportion of the document consists
    of alphanumeric or normal whitespace characters.
    """
    if not text:
        return False

    text_characters = sum(
        1
        for char in text
        if char.isalnum() or char.isspace()
    )

    return (text_characters / len(text)) >= min_ratio


def has_excessive_repetition(
    text: str,
    max_repeated_characters: int = 8,
) -> bool:
    """
    Reject obvious character spam such as '!!!!!!!!!!!!'.
    """
    pattern = rf"(.)\1{{{max_repeated_characters},}}"
    return re.search(pattern, text) is None


def passes_basic_filters(text: str) -> bool:
    """Apply all conservative document-level quality filters."""
    return (
        is_nonempty(text)
        and meets_minimum_length(text)
        and has_reasonable_text_ratio(text)
        and has_excessive_repetition(text)
    )