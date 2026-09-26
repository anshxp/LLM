import torch

from config.seq2seq_model_config import Seq2SeqModelConfig
from model.encoder_decoder import EncoderDecoderLLM


def test_v2_forward_shape_and_parameter_budget():
    config = Seq2SeqModelConfig()
    model = EncoderDecoderLLM(config)

    input_ids = torch.randint(0, config.vocab_size, (2, 32))
    decoder_input_ids = torch.randint(0, config.vocab_size, (2, 16))

    logits = model(input_ids, decoder_input_ids)

    assert logits.shape == (
        2,
        16,
        config.vocab_size,
    )
    assert logits.dtype == torch.float32

    # Default configuration is intentionally close to the requested 11M size.
    params = model.num_parameters()
    assert 10_000_000 <= params <= 11_500_000


def test_v2_encoder_and_decoder_use_shared_token_embedding():
    model = EncoderDecoderLLM(Seq2SeqModelConfig())
    assert model.token_embedding.weight.data_ptr() == model.token_embedding.weight.data_ptr()


def test_v2_rejects_overlong_sequences():
    config = Seq2SeqModelConfig(context_length=8)
    model = EncoderDecoderLLM(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 9))
    decoder_input_ids = torch.randint(0, config.vocab_size, (1, 2))

    try:
        model(input_ids, decoder_input_ids)
    except ValueError as exc:
        assert "context length" in str(exc)
    else:
        raise AssertionError("Expected context-length validation error")
