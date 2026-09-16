from config.model_config import ModelConfig
from model.llm import LLM
from tools.model_info import parameter_count


def test_parameter_count_matches_parameter_sum():
    model = LLM(ModelConfig(vocab_size=16, context_length=8, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0))
    assert parameter_count(model) == sum(p.numel() for p in model.parameters())
