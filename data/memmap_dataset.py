import mmap
import os
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


UINT32 = np.dtype("<u4")
HEADER_STRUCT = struct.Struct("<8sQ")
MAGIC = b"RXLMTOK1"
HEADER_SIZE = HEADER_STRUCT.size


class MemmapTokenStore:
    """Disk-backed uint32 token store with constant-RAM reads."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh = None
        self._tokens = None
        self.length = 0

    def open(self):
        if not self.path.exists():
            raise FileNotFoundError(f"Token store not found: {self.path}")
        self._fh = self.path.open("rb")
        raw = self._fh.read(HEADER_SIZE)
        if len(raw) != HEADER_SIZE:
            self.close()
            raise ValueError(f"Invalid token store header: {self.path}")
        magic, length = HEADER_STRUCT.unpack(raw)
        if magic != MAGIC:
            self.close()
            raise ValueError(f"Invalid token store magic: {self.path}")
        expected_size = HEADER_SIZE + length * UINT32.itemsize
        actual_size = self.path.stat().st_size
        if actual_size != expected_size:
            self.close()
            raise ValueError(
                f"Corrupt token store size for {self.path}: "
                f"expected {expected_size}, got {actual_size}"
            )
        self.length = int(length)
        self._tokens = np.memmap(
            self.path,
            dtype=UINT32,
            mode="r",
            offset=HEADER_SIZE,
            shape=(self.length,),
        )
        return self

    def close(self):
        if self._tokens is not None:
            del self._tokens
        if self._fh is not None:
            self._fh.close()
        self._tokens = None
        self._fh = None
        self.length = 0

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def slice(self, start: int, stop: int) -> np.ndarray:
        if self._tokens is None:
            raise RuntimeError("Token store is not open")
        if not (0 <= start <= stop <= self.length):
            raise IndexError("Token slice is out of bounds")
        return np.asarray(self._tokens[start:stop])


class MemmapLanguageModelDataset(Dataset):
    """Fixed-length language-model windows backed by a token memmap."""

    def __init__(self, token_store: MemmapTokenStore | str | Path, context_length: int, stride=None):
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        stride = context_length if stride is None else stride
        if stride <= 0:
            raise ValueError("stride must be positive")
        self.context_length = int(context_length)
        self.stride = int(stride)
        self._owns_store = not isinstance(token_store, MemmapTokenStore)
        self.token_store = (
            MemmapTokenStore(token_store) if self._owns_store else token_store
        )
        self.token_store.open() if self._owns_store else None

    def __del__(self):
        if getattr(self, "_owns_store", False):
            try:
                self.token_store.close()
            except Exception:
                pass

    def __len__(self):
        available = self.token_store.length - self.context_length
        if available <= 0:
            return 0
        return (available + self.stride - 1) // self.stride

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Dataset index out of range")
        start = index * self.stride
        values = self.token_store.slice(start, start + self.context_length + 1)
        if values.shape[0] != self.context_length + 1:
            raise IndexError("Incomplete training window")
        input_ids = torch.from_numpy(np.asarray(values[:-1], dtype=np.int64))
        target_ids = torch.from_numpy(np.asarray(values[1:], dtype=np.int64))
        return input_ids, target_ids
