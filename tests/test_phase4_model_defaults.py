from config.model_config import ModelConfig


def test_model_defaults_fit_small_local_model():
    config = ModelConfig()
    assert config.vocab_size == 10000
    assert config.context_length == 256
    assert config.embedding_dim == 256
    assert config.num_layers == 4
    assert config.num_heads == 4
