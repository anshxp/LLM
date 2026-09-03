from data.cleaner import (
    clean_text,
    normalize_line_endings,
    normalize_unicode,
    normalize_whitespace,
    remove_control_characters,
)


def test_normalize_line_endings():
    text = "one\r\ntwo\rthree\nfour"
    assert normalize_line_endings(text) == "one\ntwo\nthree\nfour"


def test_normalize_whitespace():
    text = "  Hello    world  \n\n\n  Next paragraph  "
    assert normalize_whitespace(text) == "Hello world\n\nNext paragraph"


def test_clean_text():
    text = "  Pneumonia  is\r\nan infection.\n\n\nThe patient was treated.  "

    assert clean_text(text) == (
        "Pneumonia is\nan infection.\n\nThe patient was treated."
    )


def test_unicode_normalization():
    text = "ＡＢＣ"
    assert normalize_unicode(text) == "ABC"


def test_control_characters_removed():
    text = "Hello\x00world\nTest"
    assert remove_control_characters(text) == "Helloworld\nTest"


def test_non_string_input():
    try:
        clean_text(None)
    except TypeError:
        pass
    else:
        raise AssertionError("clean_text should reject non-string input")