import pytest
import torch

from config.model_config import ModelConfig
from inference.generate import generate
from model.llm import LLM


def make_model(vocab_size=16, context_length=8):
    return LLM(
        ModelConfig(
            vocab_size=vocab_size,
            context_length=context_length,
            embedding_dim=32,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
        )
    )


def test_greedy_generation_is_deterministic():
    torch.manual_seed(1)
    model = make_model()
    prompt = torch.tensor([[1, 2, 3]])

    first = generate(model, prompt, max_new_tokens=5, do_sample=False)
    second = generate(model, prompt, max_new_tokens=5, do_sample=False)

    assert torch.equal(first, second)
    assert first.shape == (1, 8)


def test_top_p_generation_stays_in_vocabulary():
    torch.manual_seed(2)
    model = make_model(vocab_size=12)
    prompt = torch.tensor([[1, 2]])

    output = generate(model, prompt, max_new_tokens=5, top_p=0.8)

    assert output.shape == (1, 7)
    assert output.min().item() >= 0
    assert output.max().item() < 12


def test_top_p_one_is_valid():
    model = make_model()
    output = generate(model, torch.tensor([[1, 2]]), max_new_tokens=1, top_p=1.0)
    assert output.shape == (1, 3)


def test_invalid_top_p_is_rejected():
    model = make_model()
    with pytest.raises(ValueError):
        generate(model, torch.tensor([[1]]), max_new_tokens=1, top_p=0)
    with pytest.raises(ValueError):
        generate(model, torch.tensor([[1]]), max_new_tokens=1, top_p=1.1)


def test_eos_can_stop_generation_early():
    model = make_model(vocab_size=8)
    prompt = torch.tensor([[1, 2]])
    eos_id = 0

    # Force the model to prefer EOS at every position by replacing forward.
    def fake_forward(input_ids):
        logits = torch.full(
            (input_ids.size(0), input_ids.size(1), 8),
            -100.0,
        )
        logits[:, -1, eos_id] = 100.0
        return logits

    model.forward = fake_forward
    output = generate(
        model,
        prompt,
        max_new_tokens=10,
        do_sample=False,
        eos_token_id=eos_id,
    )

    assert output.shape == (1, 3)
    assert output[0, -1].item() == eos_id


def test_sampling_can_be_disabled_without_temperature_dependency():
    model = make_model()
    output = generate(
        model,
        torch.tensor([[1, 2]]),
        max_new_tokens=1,
        temperature=0.0,
        do_sample=False,
    )
    assert output.shape == (1, 3)
