import math

import torch

from training.loss import language_model_loss


def evaluate(model, data_loader, device=None):
    """Return mean next-token cross-entropy loss and perplexity."""
    if device is not None:
        model.to(device)

    model.eval()
    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for input_ids, target_ids in data_loader:
            if device is not None:
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)

            logits = model(input_ids)
            total_loss += language_model_loss(logits, target_ids).item()
            batches += 1

    if batches == 0:
        raise ValueError("Cannot evaluate an empty data loader")

    mean_loss = total_loss / batches
    return {"loss": mean_loss, "perplexity": math.exp(mean_loss)}
