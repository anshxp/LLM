import torch.nn.functional as F


def language_model_loss(logits, targets):
    batch_size, sequence_length, vocab_size = logits.shape

    logits = logits.view(
        batch_size * sequence_length,
        vocab_size
    )

    targets = targets.view(
        batch_size * sequence_length
    )

    loss = F.cross_entropy(
        logits,
        targets
    )

    return loss