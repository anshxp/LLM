from config.model_config import ModelConfig
from model.llm import LLM
from tools.model_info import parameter_memory_mb


def test_small_model_fp32_memory_is_reasonable():
    config = ModelConfig(vocab_size=32, context_length=8, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0)
    model = LLM(config)
    assert parameter_memory_mb(model) < 10
