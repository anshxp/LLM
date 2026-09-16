import inspect

from inference.generate import generate


def test_generation_has_context_safe_sampling_controls():
    sig = inspect.signature(generate)
    assert "max_new_tokens" in sig.parameters
    assert "temperature" in sig.parameters
    assert "top_k" in sig.parameters
