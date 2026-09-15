import torch
from torch.utils.data import Dataset


class LanguageModelDataset(Dataset):
    """Fixed-length next-token-prediction sequences over token IDs."""

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
