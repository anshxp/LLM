"""Disk-backed Parquet language-model dataset for continued pretraining."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch.utils.data import IterableDataset


TEXT_FIELDS = (
    "text",
    "content",
    "document",
    "body",
    "passage",
    "article",
    "context",
)
QUESTION_FIELDS = ("question", "prompt", "query", "user", "instruction")
ANSWER_FIELDS = ("response", "answer", "output", "completion", "assistant")


def _clean(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def row_to_text(row: dict) -> str:
    """Convert a Parquet row into plain LM text without changing the tokenizer."""
    lowered = {str(key).lower(): value for key, value in row.items()}

    for field in TEXT_FIELDS:
        value = _clean(lowered.get(field))
        if value:
            return value

    question = next((_clean(lowered.get(field)) for field in QUESTION_FIELDS if _clean(lowered.get(field))), "")
    answer = next((_clean(lowered.get(field)) for field in ANSWER_FIELDS if _clean(lowered.get(field))), "")
    if question and answer:
        return f"Question: {question}\nAnswer: {answer}"
    if question:
        return question
    if answer:
        return answer

    parts = []
    for key, value in row.items():
        if isinstance(value, (str, int, float)) and _clean(value):
            parts.append(f"{key}: {_clean(value)}")
    return "\n".join(parts)


def _split_for_row(row_index: int, row: dict) -> str:
    """Stable 90/5/5 split so train/validation/test stay deterministic."""
    identity = "\x1f".join(f"{key}={row[key]}" for key in sorted(row))
    digest = hashlib.sha256(f"{row_index}:{identity}".encode("utf-8", errors="replace")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


class ParquetLanguageModelDataset(IterableDataset):
    """Stream a Parquet corpus in bounded RAM and tokenize rows incrementally."""

    def __init__(self, parquet_file, tokenizer, context_length, stride=None, split="train"):
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if stride is None:
            stride = context_length
        if stride <= 0:
            raise ValueError("stride must be positive")
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")

        self.parquet_file = Path(parquet_file)
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.stride = stride
        self.split = split
        self.total_bytes = self.parquet_file.stat().st_size
        self.bytes_read = 0
        self.tokens_yielded = 0
        self.sequences_yielded = 0
        self.rows_seen = 0

    @property
    def total_mb(self):
        return self.total_bytes / (1024 * 1024)

    @property
    def data_consumed_mb(self):
        return self.bytes_read / (1024 * 1024)

    @property
    def data_remaining_mb(self):
        return max(0.0, (self.total_bytes - self.bytes_read) / (1024 * 1024))

    @property
    def progress_fraction(self):
        if self.total_bytes <= 0:
            return 0.0
        return min(1.0, self.bytes_read / self.total_bytes)

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
        import pyarrow.parquet as pq

        self.bytes_read = 0
        self.tokens_yielded = 0
        self.sequences_yielded = 0
        self.rows_seen = 0
        buffer = []
        next_start = 0

        parquet = pq.ParquetFile(self.parquet_file)
        total_rows = max(1, parquet.metadata.num_rows)

        for batch in parquet.iter_batches(batch_size=10000):
            rows = batch.to_pylist()
            for row in rows:
                row_index = self.rows_seen
                self.rows_seen += 1
                if _split_for_row(row_index, row) != self.split:
                    continue

                text = row_to_text(row)
                if not text:
                    continue
                token_ids = self.tokenizer.encode(text)
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

            self.bytes_read = min(
                self.total_bytes,
                int(self.total_bytes * self.rows_seen / total_rows),
            )
