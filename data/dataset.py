import torch
from torch.utils.data import Dataset, IterableDataset


class LanguageModelDataset(Dataset):
    """Fixed-length next-token-prediction sequences over an in-memory token list."""

    def __init__(self, token_ids, context_length, stride=None):
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if stride is None:
            stride = context_length
        if stride <= 0:
            raise ValueError("stride must be positive")

        self.token_ids = token_ids
        self.context_length = context_length
        self.stride = stride

    def __len__(self):
        available = len(self.token_ids) - self.context_length
        if available <= 0:
            return 0
        return (available + self.stride - 1) // self.stride

    def __getitem__(self, index):
        start = index * self.stride
        input_ids = self.token_ids[start:start + self.context_length]
        target_ids = self.token_ids[start + 1:start + self.context_length + 1]

        if len(input_ids) != self.context_length or len(target_ids) != self.context_length:
            raise IndexError("Dataset index points beyond a complete training sequence")

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )


class StreamingLanguageModelDataset(IterableDataset):
    """Disk-backed language-model dataset.

    The corpus is read incrementally and tokenized one text record at a time.
    Only the small token buffer needed to form the next training sequence is
    kept in RAM. This avoids materializing a multi-GB corpus as a Python list
    of token IDs before training.
    """

    def __init__(self, corpus_file, tokenizer, context_length, stride=None):
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if stride is None:
            stride = context_length
        if stride <= 0:
            raise ValueError("stride must be positive")

        self.corpus_file = str(corpus_file)
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.stride = stride

    def __iter__(self):
        # Keep only enough token IDs to produce the next sequence. The corpus
        # itself remains on disk, so RAM usage is independent of corpus size.
        buffer = []
        next_start = 0

        with open(self.corpus_file, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                token_ids = self.tokenizer.encode(line)
                if not token_ids:
                    continue
                buffer.extend(token_ids)

                while len(buffer) - next_start >= self.context_length + 1:
                    start = next_start
                    input_ids = buffer[start:start + self.context_length]
                    target_ids = buffer[start + 1:start + self.context_length + 1]

                    yield (
                        torch.tensor(input_ids, dtype=torch.long),
                        torch.tensor(target_ids, dtype=torch.long),
                    )
                    next_start += self.stride

                # Discard tokens that can no longer participate in a future
                # sequence. Retain the overlap required by the configured stride.
                if next_start > self.context_length * 4:
                    buffer = buffer[next_start:]
                    next_start = 0

        # Drop any incomplete tail rather than yielding a short sequence.

    def __len__(self):
        raise TypeError(
            "StreamingLanguageModelDataset has no cheap exact length; "
            "iterate it instead of calling len(dataset)."
        )
