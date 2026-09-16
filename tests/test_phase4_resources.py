from config.model_config import ModelConfig
from model.llm import LLM
from tools.model_info import parameter_memory_mb


def test_resource_estimate_is_positive():
    model = LLM(ModelConfig(vocab_size=16, context_length=8, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0))
    assert parameter_memory_mb(model) > 0
