import torch.nn.functional as F


def language_model_loss(logits, targets):
    """Cross-entropy language-model loss with support for masked targets.

    Instruction SFT uses -100 for prompt and padding positions. Those positions
    must not contribute to the optimization objective.
    """
    batch_size, sequence_length, vocab_size = logits.shape

    logits = logits.reshape(batch_size * sequence_length, vocab_size)
    targets = targets.reshape(batch_size * sequence_length)

    return F.cross_entropy(logits, targets, ignore_index=-100)
