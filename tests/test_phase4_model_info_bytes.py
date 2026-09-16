from config.model_config import ModelConfig
from model.llm import LLM
from tools.model_info import parameter_memory_mb


def test_memory_estimate_scales_with_precision():
    model = LLM(ModelConfig(vocab_size=16, context_length=8, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0))
    fp32 = parameter_memory_mb(model, 4)
    fp16 = parameter_memory_mb(model, 2)
    assert fp16 == fp32 / 2
