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
    """Disk-backed LM dataset with bounded RAM and observable byte progress."""

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
        self.total_bytes = self._get_total_bytes()
        self.bytes_read = 0
        self.tokens_yielded = 0
        self.sequences_yielded = 0

    def _get_total_bytes(self):
        try:
            return max(0, int(__import__("os").path.getsize(self.corpus_file)))
        except OSError:
            return 0

    @property
    def progress_fraction(self):
        if self.total_bytes <= 0:
            return 0.0
        return min(1.0, self.bytes_read / self.total_bytes)

    @property
    def data_consumed_mb(self):
        return self.bytes_read / (1024 * 1024)

    @property
    def data_remaining_mb(self):
        return max(0.0, (self.total_bytes - self.bytes_read) / (1024 * 1024))

    @property
    def total_mb(self):
        return self.total_bytes / (1024 * 1024)

    def progress_snapshot(self):
        return {
            "bytes_read": self.bytes_read,
            "total_bytes": self.total_bytes,
            "data_consumed_mb": self.data_consumed_mb,
            "data_remaining_mb": self.data_remaining_mb,
            "progress_fraction": self.progress_fraction,
            "tokens_yielded": self.tokens_yielded,
            "sequences_yielded": self.sequences_yielded,
        }

    def __iter__(self):
        buffer = []
        next_start = 0
        self.bytes_read = 0
        self.tokens_yielded = 0
        self.sequences_yielded = 0

        # Binary iteration gives an exact byte position for progress logging,
        # while decoding each line keeps tokenizer input as normal text.
        with open(self.corpus_file, "rb") as handle:
            for raw_line in handle:
                self.bytes_read += len(raw_line)
                line = raw_line.decode("utf-8", errors="replace")
                token_ids = self.tokenizer.encode(line)
                if not token_ids:
                    continue
                buffer.extend(token_ids)

                while len(buffer) - next_start >= self.context_length + 1:
                    start = next_start
                    input_ids = buffer[start:start + self.context_length]
                    target_ids = buffer[start + 1:start + self.context_length + 1]
                    self.tokens_yielded += self.context_length
                    self.sequences_yielded += 1
                    yield (
                        torch.tensor(input_ids, dtype=torch.long),
                        torch.tensor(target_ids, dtype=torch.long),
                    )
                    next_start += self.stride

                if next_start > self.context_length * 4:
                    buffer = buffer[next_start:]
                    next_start = 0

    def __len__(self):
        raise TypeError(
            "StreamingLanguageModelDataset has no cheap exact length; "
            "iterate it instead of calling len(dataset)."
        )
