import pytest
import torch

from inference.generate import generate


class TinyModel(torch.nn.Module):
    def __init__(self, vocab_size=8, context_length=4, preferred_token=3):
        super().__init__()
        self.embedding = torch.nn.Module()
        self.embedding.position_embedding = torch.nn.Embedding(context_length, 2)
        self.vocab_size = vocab_size
        self.preferred_token = preferred_token
        self.calls = []

    def forward(self, input_ids):
        self.calls.append(input_ids.detach().clone())
        logits = torch.full(
            (input_ids.size(0), input_ids.size(1), self.vocab_size),
            -10.0,
            device=input_ids.device,
        )
        logits[..., self.preferred_token] = 10.0
        return logits


def test_greedy_generation_is_deterministic_and_appends_requested_tokens():
    model = TinyModel(preferred_token=3)
    input_ids = torch.tensor([[1, 2]])

    output = generate(
        model,
        input_ids,
        max_new_tokens=3,
        do_sample=False,
    )

    assert output.tolist() == [[1, 2, 3, 3, 3]]
    assert len(model.calls) == 3


def test_generation_limits_model_context_to_position_embedding_capacity():
    model = TinyModel(context_length=4, preferred_token=3)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    output = generate(model, input_ids, max_new_tokens=2, do_sample=False)

    assert output.shape == (1, 7)
    assert all(call.shape[1] <= 4 for call in model.calls)
    assert model.calls[0].tolist() == [[2, 3, 4, 5]]
    assert model.calls[1].tolist() == [[3, 4, 5, 3]]


def test_eos_stops_generation_for_finished_batch():
    model = TinyModel(preferred_token=2)
    input_ids = torch.tensor([[1]])

    output = generate(
        model,
        input_ids,
        max_new_tokens=10,
        do_sample=False,
        eos_token_id=2,
    )

    assert output.tolist() == [[1, 2]]
    assert len(model.calls) == 1


def test_zero_new_tokens_returns_input_without_model_call():
    model = TinyModel()
    input_ids = torch.tensor([[1, 2]])

    output = generate(model, input_ids, max_new_tokens=0, do_sample=False)

    assert torch.equal(output, input_ids)
    assert model.calls == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_new_tokens": -1},
        {"max_new_tokens": 1, "temperature": 0},
        {"max_new_tokens": 1, "top_k": 0},
        {"max_new_tokens": 1, "top_p": 0},
        {"max_new_tokens": 1, "top_p": 1.1},
        {"max_new_tokens": 1, "eos_token_id": -1},
    ],
)
def test_generation_rejects_invalid_decoding_arguments(kwargs):
    model = TinyModel()
    input_ids = torch.tensor([[1]])

    with pytest.raises(ValueError):
        generate(model, input_ids, do_sample=True, **kwargs)


def test_generation_rejects_non_matrix_input():
    model = TinyModel()

    with pytest.raises(ValueError, match=r"shape \[batch, sequence\]"):
        generate(model, torch.tensor([1, 2]), max_new_tokens=1, do_sample=False)
