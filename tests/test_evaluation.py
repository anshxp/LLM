import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluation.evaluate import evaluate


class ConstantModel(torch.nn.Module):
    def __init__(self, vocab_size, preferred_token):
        super().__init__()
        self.vocab_size = vocab_size
        self.preferred_token = preferred_token

    def forward(self, input_ids):
        batch, sequence = input_ids.shape
        logits = torch.zeros(batch, sequence, self.vocab_size)
        logits[..., self.preferred_token] = 2.0
        return logits


def test_evaluate_returns_loss_and_perplexity():
    inputs = torch.tensor([[0, 1, 2], [1, 2, 0]])
    targets = torch.tensor([[2, 2, 2], [2, 2, 2]])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    metrics = evaluate(ConstantModel(4, 2), loader)

    assert set(metrics) == {"loss", "perplexity"}
    assert metrics["loss"] > 0
    assert math.isclose(metrics["perplexity"], math.exp(metrics["loss"]), rel_tol=1e-6)
