from data.dedup import hamming_distance, is_near_duplicate, simhash


def test_simhash_is_deterministic():
    text = "Pneumonia is an infection of the lungs."
    assert simhash(text) == simhash(text)


def test_simhash_detects_similar_text():
    first = simhash("Pneumonia is an infection of the lungs and may cause cough and fever.")
    second = simhash("Pneumonia is an infection of the lungs and may cause cough and fever. It is common.")
    assert hamming_distance(first, second) <= 12
    assert is_near_duplicate(second, [first])


def test_simhash_separates_different_text():
    first = simhash("Pneumonia is an infection of the lungs and may cause cough and fever.")
    second = simhash("The stock market closed higher after technology companies reported earnings.")
    assert hamming_distance(first, second) > 12
    assert not is_near_duplicate(second, [first])
