from data.filters import (
    has_excessive_repetition,
    has_reasonable_text_ratio,
    is_nonempty,
    meets_minimum_length,
    passes_basic_filters,
)


def test_is_nonempty():
    assert is_nonempty("Medical text")
    assert not is_nonempty("   ")


def test_meets_minimum_length():
    assert meets_minimum_length("a" * 200)
    assert not meets_minimum_length("a" * 199)


def test_has_reasonable_text_ratio():
    assert has_reasonable_text_ratio(
        "Pneumonia is a common respiratory infection."
    )
    assert not has_reasonable_text_ratio("!!!!!!@@@@####$$$$%%%%")


def test_has_excessive_repetition():
    assert has_excessive_repetition("The patient has pneumonia.")
    assert not has_excessive_repetition("AAAAAAAAAAAA")


def test_passes_basic_filters():
    valid_text = (
        "Pneumonia is an infection of the lungs. "
        "It may be caused by bacteria, viruses, or other microorganisms. "
        "Clinical presentation can vary depending on the underlying cause. "
        "Patients may develop fever, cough, shortness of breath, chest pain, "
        "and fatigue. Diagnosis commonly involves clinical assessment, "
        "laboratory testing, and imaging when appropriate."
    )

    assert passes_basic_filters(valid_text)


def test_rejects_empty_text():
    assert not passes_basic_filters("")


def test_rejects_short_text():
    assert not passes_basic_filters("Too short.")


def test_rejects_repetitive_text():
    text = ("A" * 10) + " medical text " * 30
    assert not passes_basic_filters(text)