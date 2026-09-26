"""Streaming datasets for the v2 encoder-decoder model."""

import json
from pathlib import Path

import torch
from torch.utils.data import IterableDataset

from data.instruction_dataset import format_example
from data.tokenizer import Tokenizer

DEFAULT_TOKENIZER = Path("data/processed/tokenizer.json")


class BookSeq2SeqDataset(IterableDataset):
    """Turn book text into prefix-to-continuation training examples."""

    def __init__(self, corpus_file, tokenizer_path=DEFAULT_TOKENIZER, context_length=256, stride=None):
        self.corpus_file = Path(corpus_file)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.context_length = context_length
        self.source_length = context_length // 2
        self.target_length = context_length - self.source_length
        self.stride = stride or self.source_length
        if self.source_length < 2 or self.target_length < 2:
            raise ValueError("context_length must be at least 4")
        if not self.corpus_file.exists():
            raise FileNotFoundError(f"Book corpus not found: {self.corpus_file}")

    def _paragraphs(self):
        buffer = []
        with self.corpus_file.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.strip():
                    buffer.append(line)
                elif buffer:
                    yield "".join(buffer)
                    buffer.clear()
            if buffer:
                yield "".join(buffer)

    def __iter__(self):
        bos = self.tokenizer.token_to_id["<bos>"]
        for paragraph in self._paragraphs():
            token_ids = self.tokenizer.encode(paragraph, add_bos=True, add_eos=True)
            if len(token_ids) < self.source_length + self.target_length:
                continue
            for start in range(0, len(token_ids) - self.source_length - 1, self.stride):
                source = token_ids[start:start + self.source_length]
                target = token_ids[start + self.source_length:start + self.source_length + self.target_length]
                if len(source) != self.source_length or len(target) < 2:
                    continue
                yield _make_example(source, target, bos)


class InstructionSeq2SeqDataset(IterableDataset):
    """Streaming instruction dataset: encoder=prompt, decoder=response."""

    def __init__(self, jsonl_file_or_dir, tokenizer_path=DEFAULT_TOKENIZER, context_length=256):
        path = Path(jsonl_file_or_dir)
        if path.is_dir():
            self.files = sorted(path.glob("*.jsonl"))
        elif path.exists():
            self.files = [path]
        else:
            self.files = sorted(path.parent.glob(f"{path.stem}-*.jsonl"))
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.context_length = context_length
        if not self.files:
            raise FileNotFoundError(f"No SFT JSONL shards found for {path}")

    def __iter__(self):
        bos = self.tokenizer.token_to_id["<bos>"]
        for jsonl_file in self.files:
            with jsonl_file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    prompt, response = format_example(record)
                    source = self.tokenizer.encode(prompt, add_bos=True, add_eos=True)
                    target = self.tokenizer.encode(response, add_bos=False, add_eos=True)
                    if not target:
                        continue
                    source = source[-self.context_length:]
                    target = target[: self.context_length - 1]
                    if not source or not target:
                        continue
                    decoder_input = [bos] + target[:-1]
                    yield {
                        "encoder_input_ids": torch.tensor(source, dtype=torch.long),
                        "decoder_input_ids": torch.tensor(decoder_input, dtype=torch.long),
                        "labels": torch.tensor(target, dtype=torch.long),
                    }


def _make_example(source, target, bos_id):
    decoder_input = [bos_id] + target[:-1]
    return {
        "encoder_input_ids": torch.tensor(source, dtype=torch.long),
        "decoder_input_ids": torch.tensor(decoder_input, dtype=torch.long),
        "labels": torch.tensor(target, dtype=torch.long),
    }


def collate_seq2seq(batch, pad_id=0):
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    max_source = max(item["encoder_input_ids"].numel() for item in batch)
    max_target = max(item["decoder_input_ids"].numel() for item in batch)
    encoder = torch.full((len(batch), max_source), pad_id, dtype=torch.long)
    decoder = torch.full((len(batch), max_target), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_target), -100, dtype=torch.long)
    for i, item in enumerate(batch):
        s = item["encoder_input_ids"]
        d = item["decoder_input_ids"]
        y = item["labels"]
        encoder[i, :len(s)] = s
        decoder[i, :len(d)] = d
        labels[i, :len(y)] = y
    return encoder, decoder, labels
