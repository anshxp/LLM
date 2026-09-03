import torch

from training.loss import language_model_loss


def train_step(model, optimizer, input_ids, target_ids):
    model.train()

    optimizer.zero_grad()

    logits = model(input_ids)

    loss = language_model_loss(
        logits,
        target_ids,
    )

    loss.backward()

    optimizer.step()

    return loss.item()