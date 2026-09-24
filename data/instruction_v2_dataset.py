"""Lazy dataset for the sharded second-fine-tuning corpus."""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from data.instruction_dataset import collate_instruction_batch, format_example
from data.tokenizer import Tokenizer


class ShardedInstructionDataset(Dataset):
    """Random-access, lazy-tokenized dataset backed by JSONL shards.

    Only byte offsets are retained in memory; records and token tensors are created
    when an item is requested. This avoids loading millions of SFT examples into RAM.
    """

    def __init__(self, directory, split, context_length, tokenizer_path="data/processed/tokenizer.json", categories=None):
        self.directory = Path(directory)
        self.split = split
        self.context_length = context_length
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.categories = None if categories is None else set(categories)
        self.locations = []
        self._handles = {}

        paths = sorted(self.directory.glob(f"{split}-*.jsonl"))
        if not paths:
            raise FileNotFoundError(f"No {split}-*.jsonl shards found in {self.directory}")
        for path in paths:
            with path.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    if self.categories is not None:
                        import json
                        record = json.loads(line)
                        if record.get("category") not in self.categories:
                            continue
                    self.locations.append((path, offset))

    def _read_record(self, index):
        import json
        path, offset = self.locations[index]
        handle = self._handles.get(path)
        if handle is None:
            handle = path.open("rb")
            self._handles[path] = handle
        handle.seek(offset)
        return json.loads(handle.readline())

    def _encode(self, record):
        prompt, response = format_example(record)
        prompt_ids = self.tokenizer.encode(prompt, add_bos=True)
        response_ids = self.tokenizer.encode(response, add_eos=True)
        max_tokens = self.context_length + 1
        if len(response_ids) >= max_tokens:
            response_ids = response_ids[-max_tokens:]
            prompt_ids = []
        else:
            prompt_ids = prompt_ids[-(max_tokens - len(response_ids)):]
        token_ids = prompt_ids + response_ids
        if len(token_ids) < 2:
            raise ValueError("SFT example must contain at least two tokens")
        inputs = torch.tensor(token_ids[:-1], dtype=torch.long)
        labels = torch.tensor(token_ids[1:], dtype=torch.long)
        labels[: max(0, len(prompt_ids) - 1)] = -100
        return inputs, labels

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, index):
        return self._encode(self._read_record(index))

    def __del__(self):
        for handle in getattr(self, "_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass


__all__ = ["ShardedInstructionDataset", "collate_instruction_batch"]
