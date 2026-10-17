import hashlib
from pathlib import Path

import torch
from torch.utils.data import IterableDataset

from data.parquet_dataset import row_to_text, _split_for_row


class ParquetSeq2SeqDataset(IterableDataset):
    """Stream text and make source-prefix -> future-continuation training pairs.

    Each example has a bidirectional encoder source consisting of context_length
    tokens and a decoder target consisting of the next context_length tokens.
    Rows are split deterministically before tokenization and split buffers never
    mix train/validation/test records. EOS is inserted between records.
    """

    def __init__(self, parquet_file, tokenizer, context_length=512, split="train"):
        if context_length < 2:
            raise ValueError("context_length must be >= 2")
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        self.parquet_file = Path(parquet_file)
        if not self.parquet_file.exists():
            raise FileNotFoundError(f"Parquet corpus not found: {self.parquet_file}")
        self.tokenizer = tokenizer
        self.context_length = int(context_length)
        self.split = split
        self.total_bytes = self.parquet_file.stat().st_size
        self.bytes_read = 0
        self.tokens_seen = 0
        self.sequences_yielded = 0
        self.bos_id = tokenizer.token_to_id["<bos>"]
        self.eos_id = tokenizer.token_to_id["<eos>"]
        self.pad_id = tokenizer.token_to_id["<pad>"]

    @property
    def total_mb(self):
        return self.total_bytes / (1024 * 1024)

    @property
    def progress_fraction(self):
        return 0.0 if not self.total_bytes else min(1.0, self.bytes_read / self.total_bytes)

    @property
    def data_consumed_mb(self):
        return self.bytes_read / (1024 * 1024)

    @property
    def data_remaining_mb(self):
        return max(0.0, (self.total_bytes - self.bytes_read) / (1024 * 1024))

    def progress_snapshot(self):
        return {
            "total_bytes": self.total_bytes,
            "bytes_read": self.bytes_read,
            "data_consumed_mb": self.data_consumed_mb,
            "data_remaining_mb": self.data_remaining_mb,
            "progress_fraction": self.progress_fraction,
            "tokens_seen": self.tokens_seen,
            "sequences_yielded": self.sequences_yielded,
        }

    def __iter__(self):
        import pyarrow.parquet as pq

        # Build a stream only from rows assigned to this split. EOS keeps row
        # boundaries visible; examples may span a boundary but never a split.
        tokens = []
        self.bytes_read = self.tokens_seen = self.sequences_yielded = 0
        parquet = pq.ParquetFile(self.parquet_file)
        total_rows = max(1, parquet.metadata.num_rows)
        rows_seen = 0
        pair_width = 2 * self.context_length
        next_start = 0

        for batch in parquet.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                row_index = rows_seen
                rows_seen += 1
                if _split_for_row(row_index, row) != self.split:
                    continue
                text = row_to_text(row)
                if not text:
                    continue
                row_tokens = self.tokenizer.encode(text)
                if not row_tokens:
                    continue
                tokens.extend(row_tokens)
                tokens.append(self.eos_id)

                while len(tokens) - next_start >= pair_width:
                    start = next_start
                    source = tokens[start:start + self.context_length]
                    target = tokens[start + self.context_length:start + pair_width]
                    decoder_input = [self.bos_id] + target[:-1]
                    self.tokens_seen += pair_width
                    self.sequences_yielded += 1
                    yield (
                        torch.tensor(source, dtype=torch.long),
                        torch.tensor(decoder_input, dtype=torch.long),
                        torch.tensor(target, dtype=torch.long),
                    )
                    next_start += self.context_length

                if next_start > 4 * pair_width:
                    tokens = tokens[next_start:]
                    next_start = 0

            self.bytes_read = min(
                self.total_bytes,
                int(self.total_bytes * rows_seen / total_rows),
            )
