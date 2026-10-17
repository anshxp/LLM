"""Unit and smoke tests for encoder-decoder pretraining."""
import torch

from config.model_config import ModelConfig
from model.seq2seq_lm import Seq2SeqLanguageModel


def test_seq2seq_forward_and_parameter_tying():
    config = ModelConfig(vocab_size=128, context_length=32, embedding_dim=32,
                         num_encoder_layers=2, num_decoder_layers=2,
                         num_heads=4, dropout=0.0)
    model = Seq2SeqLanguageModel(config)
    source = torch.randint(0, config.vocab_size, (2, 32))
    decoder = torch.randint(0, config.vocab_size, (2, 32))
    logits = model(source, decoder)
    assert logits.shape == (2, 32, config.vocab_size)
    assert torch.isfinite(logits).all()
    assert model.lm_head.weight.data_ptr() == model.embedding.token_embedding.weight.data_ptr()


def test_source_future_tokens_can_influence_decoder_via_cross_attention():
    config = ModelConfig(vocab_size=32, context_length=8, embedding_dim=32,
                         num_encoder_layers=1, num_decoder_layers=1,
                         num_heads=4, dropout=0.0)
    model = Seq2SeqLanguageModel(config).eval()
    source = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    changed = source.clone()
    changed[0, -1] = 9
    decoder = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
    with torch.no_grad():
        a = model(source, decoder)
        b = model(changed, decoder)
    assert not torch.allclose(a[:, :4], b[:, :4])


def test_generation_returns_new_tokens_only():
    config = ModelConfig(vocab_size=32, context_length=8, embedding_dim=32,
                         num_encoder_layers=1, num_decoder_layers=1,
                         num_heads=4, dropout=0.0)
    model = Seq2SeqLanguageModel(config).eval()
    source = torch.tensor([[1, 2, 3, 4]])
    torch.manual_seed(3)
    output = model.generate(source, bos_token_id=2, max_new_tokens=3, top_k=4)
    assert output.shape == (1, 3)
    assert output.dtype == torch.long
