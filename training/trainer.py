import torch

from training.loss import language_model_loss


def train_step(model, optimizer, input_ids, target_ids, max_grad_norm=None):
    """Run one optimizer update and return the scalar training loss."""
    model.train()

    optimizer.zero_grad(set_to_none=True)

    logits = model(input_ids)
    loss = language_model_loss(logits, target_ids)
    loss.backward()

    if max_grad_norm is not None:
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive when provided")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()
    return loss.item()
