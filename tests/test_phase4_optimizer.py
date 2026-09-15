from config.model_config import ModelConfig
from model.llm import LLM
from training.optimizer import create_optimizer


def test_optimizer_contains_model_parameters():
    model = LLM(ModelConfig(vocab_size=16, context_length=8, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0))
    optimizer = create_optimizer(model, 1e-3, 0.01)
    assert sum(len(group["params"]) for group in optimizer.param_groups) == len(list(model.parameters()))
