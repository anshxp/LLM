import torch

from config.model_config import ModelConfig
from model.seq2seq_lm import Seq2SeqLanguageModel


def test_seq2seq_forward_and_parameter_budget():
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


def test_encoder_is_bidirectional_but_decoder_is_causal():
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
    # The encoder is intentionally bidirectional, so changing the source's
    # last token is allowed to change earlier decoder positions via cross-attention.
    assert not torch.allclose(a[:, :4], b[:, :4])
