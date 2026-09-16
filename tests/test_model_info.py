from config.model_config import ModelConfig
from model.llm import LLM
from tools.model_info import parameter_count, parameter_memory_mb


def test_parameter_count_is_positive():
    model = LLM(ModelConfig(vocab_size=32, context_length=8, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0))
    assert parameter_count(model) > 0


def test_parameter_memory_estimate_uses_bytes_per_parameter():
    model = LLM(ModelConfig(vocab_size=32, context_length=8, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0))
    count = parameter_count(model)
    assert parameter_memory_mb(model, bytes_per_parameter=4) == count * 4 / (1024 ** 2)
