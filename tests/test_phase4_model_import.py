from model.llm import LLM


def test_llm_imports():
    assert callable(LLM)
