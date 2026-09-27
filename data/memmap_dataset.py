"""Memory-mapped token datasets for large corpora."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

TOKEN_DTYPE = np.dtype("<u4")
HEADER_STRUCT = struct.Struct("<8sQ")
MAGIC = b"RXLMTOK1"
HEADER_SIZE = HEADER_STRUCT.size


class MemmapTokenStore:
    """Read a disk-backed token array without loading it into RAM."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._map = None
        self.length = 0

    def open(self) -> "MemmapTokenStore":
        if self._map is not None:
            return self

        with self.path.open("rb") as handle:
            raw = handle.read(HEADER_SIZE)

        if len(raw) != HEADER_SIZE:
            raise ValueError(f"Invalid token store header: {self.path}")

        magic, length = HEADER_STRUCT.unpack(raw)
        if magic != MAGIC:
            raise ValueError(f"Invalid token store magic: {self.path}")

        length = int(length)
        expected_size = HEADER_SIZE + length * TOKEN_DTYPE.itemsize
        actual_size = self.path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"Corrupt token store size for {self.path}: "
                f"expected {expected_size}, got {actual_size}"
            )

        self.length = length
        self._map = np.memmap(
            self.path,
            dtype=TOKEN_DTYPE,
            mode="r",
            offset=HEADER_SIZE,
            shape=(self.length,),
        )
        return self

    def close(self) -> None:
        if self._map is not None:
            mmap_obj = getattr(self._map, "_mmap", None)
            del self._map
            if mmap_obj is not None:
                mmap_obj.close()
        self._map = None
        self.length = 0

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def slice(self, start: int, stop: int) -> np.ndarray:
        if self._map is None:
            self.open()
        if not 0 <= start <= stop <= self.length:
            raise IndexError("Token slice is out of bounds")
        return np.asarray(self._map[start:stop])


class MemmapLanguageModelDataset(Dataset):
    """Fixed-length next-token windows backed by a token memmap."""

    def __init__(
        self,
        token_store: MemmapTokenStore | str | Path,
        context_length: int,
        stride: int | None = None,
    ):
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        stride = context_length if stride is None else stride
        if stride <= 0:
            raise ValueError("stride must be positive")

        self.context_length = int(context_length)
        self.stride = int(stride)
        if isinstance(token_store, MemmapTokenStore):
            self.token_store = token_store
            self._owns_store = False
        else:
            self.token_store = MemmapTokenStore(token_store)
            self._owns_store = True
            self.token_store.open()

    def close(self) -> None:
        if self._owns_store:
            self.token_store.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self):
        available = self.token_store.length - self.context_length
        if available <= 0:
            return 0
        return (available + self.stride - 1) // self.stride

    def __getitem__(self, index):
        size = len(self)
        if index < 0:
            index += size
        if index < 0 or index >= size:
            raise IndexError("Dataset index out of range")

        start = index * self.stride
        values = self.token_store.slice(
            start,
            start + self.context_length + 1,
        )
        if values.shape[0] != self.context_length + 1:
            raise IndexError("Incomplete training window")

        # One small window is copied into PyTorch tensors; the corpus is not.
        input_ids = torch.from_numpy(values[:-1].astype(np.int64, copy=True))
        target_ids = torch.from_numpy(values[1:].astype(np.int64, copy=True))
        return input_ids, target_ids
