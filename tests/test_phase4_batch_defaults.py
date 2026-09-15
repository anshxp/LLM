from train import DEFAULT_BATCH_SIZE, DEFAULT_GRADIENT_ACCUMULATION_STEPS


def test_memory_safe_defaults():
    assert DEFAULT_BATCH_SIZE == 1
    assert DEFAULT_GRADIENT_ACCUMULATION_STEPS == 4
