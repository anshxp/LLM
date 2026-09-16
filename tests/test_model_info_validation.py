import pytest

from config.model_config import ModelConfig
from model.llm import LLM
from tools.model_info import parameter_memory_mb


def test_parameter_memory_rejects_invalid_byte_size():
    model = LLM(ModelConfig(vocab_size=16, context_length=8, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0))
    with pytest.raises(ValueError):
        parameter_memory_mb(model, bytes_per_parameter=0)
