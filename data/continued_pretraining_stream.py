"""Streaming data sources for shard-by-shard continued pretraining.

The pipeline is intentionally bounded-memory:
- local files/directories are consumed incrementally;
- Hugging Face Parquet shards are downloaded one at a time;
- Parquet is read with pyarrow batches;
- token IDs are accumulated only up to one model context window;
- exact deduplication is persisted in SQLite instead of RAM.

The same trained tokenizer is reused for every shard.
"""
from __future__ import annotations

import csv
import fnmatch
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Iterator

import pyarrow.parquet as pq
from torch.utils.data import IterableDataset

from data.cleaner import clean_text
from data.filters import passes_basic_filters
from data.pdf_extractor import extract_pdf_text
from data.tokenizer import Tokenizer

SUPPORTED_LOCAL = {
    ".txt", ".text", ".md", ".markdown",
    ".json", ".jsonl", ".csv", ".tsv", ".xml", ".pdf", ".xlsx", ".parquet",
}
TEXT_KEYS = {"text", "content", "body", "document", "article"}
QUESTION_KEYS = {"question", "prompt", "instruction", "query", "user"}
ANSWER_KEYS = {"answer", "response", "output", "completion", "assistant"}
SKIP_KEYS = {"id", "source", "url", "metadata", "index", "uuid"}
DEFAULT_PARQUET_BATCH_SIZE = 4096


def _normalise_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def record_to_text(record: object) -> str:
    """Convert a generic structured record into deterministic training text."""
    if not isinstance(record, dict):
        return _normalise_value(record).strip()

    question: list[str] = []
    answer: list[str] = []
    text: list[str] = []
    other: list[str] = []

    for key, value in record.items():
        key_lower = str(key).lower()
        if value is None or key_lower in SKIP_KEYS:
            continue
        value_text = _normalise_value(value).strip()
        if not value_text:
            continue
        if key_lower in QUESTION_KEYS:
            question.append(value_text)
        elif key_lower in ANSWER_KEYS:
            answer.append(value_text)
        elif key_lower in TEXT_KEYS:
            text.append(value_text)
        else:
            other.append(f"{key}: {value_text}")

    if question or answer:
        parts = []
        if question:
            parts.append("Question: " + "\n".join(question))
        if answer:
            parts.append("Answer: " + "\n".join(answer))
        return "\n".join(parts)
    if text:
        return "\n".join(text)
    return "\n".join(other)


def _repair_and_clean(text: str) -> str:
    if "â" in text or "Ã" in text or "Â" in text:
        try:
            repaired = text.encode("latin1").decode("utf-8")
            if sum(ord(c) > 127 for c in repaired) >= sum(ord(c) > 127 for c in text):
                text = repaired
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return clean_text(text)


def iter_text_file(path: Path, max_chars: int = 2_000_000) -> Iterator[str]:
    """Yield bounded chunks from a text file without reading it all into RAM."""
    buffer: list[str] = []
    chars = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                if buffer:
                    yield "".join(buffer)
                    buffer.clear()
                    chars = 0
                continue
            buffer.append(line)
            chars += len(line)
            if chars >= max_chars:
                yield "".join(buffer)
                buffer.clear()
                chars = 0
    if buffer:
        yield "".join(buffer)


def iter_parquet_records(path: Path, batch_size: int = DEFAULT_PARQUET_BATCH_SIZE) -> Iterator[object]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            yield row


def iter_json_records(path: Path) -> Iterator[object]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return

    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(payload, list):
        yield from payload
    elif isinstance(payload, dict):
        for key in ("data", "records", "examples", "items"):
            if isinstance(payload.get(key), list):
                yield from payload[key]
                return
        yield payload
    else:
        yield payload


def iter_delimited_records(path: Path) -> Iterator[object]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        yield from csv.DictReader(handle, delimiter=delimiter)


def iter_xlsx_records(path: Path) -> Iterator[object]:
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        for sheet in workbook.worksheets:
            rows = sheet.iter_rows(values_only=True)
            try:
                raw_headers = next(rows)
            except StopIteration:
                continue
            headers = [str(value).strip() if value is not None else "" for value in raw_headers]
            for values in rows:
                yield {
                    headers[i]: values[i]
                    for i in range(min(len(headers), len(values)))
                    if headers[i]
                }
    finally:
        workbook.close()


def iter_xml_records(path: Path) -> Iterator[str]:
    root = ET.parse(path).getroot()
    pairs = list(root.iter("QAPair"))
    if pairs:
        for pair in pairs:
            question = (pair.findtext("Question") or "").strip()
            answer = (pair.findtext("Answer") or "").strip()
            if question and answer:
                yield f"Question: {question}\nAnswer: {answer}"
        return
    text = " ".join(t.strip() for t in root.itertext() if t.strip())
    if text:
        yield text


def iter_local_texts(path: Path, parquet_batch_size: int = DEFAULT_PARQUET_BATCH_SIZE) -> Iterator[str]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".text", ".md", ".markdown"}:
        yield from iter_text_file(path)
    elif suffix == ".parquet":
        for row in iter_parquet_records(path, batch_size=parquet_batch_size):
            text = record_to_text(row)
            if text:
                yield text
    elif suffix in {".json", ".jsonl"}:
        for row in iter_json_records(path):
            text = record_to_text(row)
            if text:
                yield text
    elif suffix in {".csv", ".tsv"}:
        for row in iter_delimited_records(path):
            text = record_to_text(row)
            if text:
                yield text
    elif suffix == ".xlsx":
        for row in iter_xlsx_records(path):
            text = record_to_text(row)
            if text:
                yield text
    elif suffix == ".xml":
        yield from iter_xml_records(path)
    elif suffix == ".pdf":
        text = extract_pdf_text(path)
        for part in text.split("\n\n"):
            if part.strip():
                yield part


def iter_local_shards(path: Path) -> Iterator[tuple[str, Path]]:
    """Yield local files in stable order; a directory becomes file-sized shards."""
    if not path.exists():
        raise FileNotFoundError(f"Local shard path not found: {path}")
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_LOCAL:
            raise ValueError(f"Unsupported local shard format: {path.suffix}")
        yield path.as_posix(), path
        return
    for child in sorted(path.rglob("*")):
        if child.is_file() and child.suffix.lower() in SUPPORTED_LOCAL:
            yield child.as_posix(), child


def hf_parquet_files(
    repo_id: str,
    repo_type: str = "dataset",
    revision: str | None = None,
    pattern: str = "*.parquet",
) -> list[str]:
    """List Parquet files on the Hub without downloading them."""
    from huggingface_hub import list_repo_files

    files = list_repo_files(repo_id, repo_type=repo_type, revision=revision)
    return sorted(name for name in files if fnmatch.fnmatch(name, pattern))


def download_hf_shard(
    repo_id: str,
    filename: str,
    download_dir: Path,
    repo_type: str = "dataset",
    revision: str | None = None,
) -> Path:
    """Download exactly one Hub file into a disposable directory."""
    from huggingface_hub import hf_hub_download

    download_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        revision=revision,
        local_dir=download_dir,
        local_dir_use_symlinks=False,
    )
    return Path(path)


class ExactDedupStore:
    """Disk-backed exact deduplication so 57GB never requires a RAM-sized set."""

    def __init__(self, path: Path):
        import sqlite3

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS seen (content_sha256 BLOB PRIMARY KEY)"
        )
        self.connection.commit()

    def add_if_new(self, text: str) -> bool:
        digest = hashlib.sha256(" ".join(text.split()).encode("utf-8")).digest()
        inserted = self.connection.execute(
            "INSERT OR IGNORE INTO seen(content_sha256) VALUES (?)", (digest,)
        ).rowcount
        return bool(inserted)

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


def iter_training_texts(
    texts: Iterable[str],
    dedup: ExactDedupStore | None = None,
    split: str = "train",
    validation_mod: int = 20,
) -> Iterator[str]:
    """Clean/filter/dedup texts and deterministically select one split."""
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")

    for raw in texts:
        cleaned = _repair_and_clean(raw)
        if not cleaned or not passes_basic_filters(cleaned):
            continue
        digest = hashlib.sha256(cleaned.encode("utf-8")).digest()
        is_validation = int.from_bytes(digest[:4], "big") % validation_mod == 0
        if (split == "validation") != is_validation:
            continue
        if dedup is not None and not dedup.add_if_new(cleaned):
            continue
        yield cleaned


class StreamingTokenDataset(IterableDataset):
    """Iterable token dataset backed by a fresh text iterator for each pass."""

    def __init__(
        self,
        text_factory,
        tokenizer: Tokenizer,
        context_length: int,
        split: str,
        eos_between_documents: bool = True,
    ):
        super().__init__()
        self.text_factory = text_factory
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.split = split
        self.eos_id = tokenizer.token_to_id["<eos>"]
        self.eos_between_documents = eos_between_documents

    def __iter__(self):
        import torch

        token_buffer: list[int] = []
        for text in self.text_factory(self.split):
            token_buffer.extend(self.tokenizer.encode(text, add_bos=True))
            if self.eos_between_documents:
                token_buffer.append(self.eos_id)

            while len(token_buffer) >= self.context_length + 1:
                window = token_buffer[: self.context_length + 1]
                del token_buffer[: self.context_length]
                yield (
                    torch.tensor(window[:-1], dtype=torch.long),
                    torch.tensor(window[1:], dtype=torch.long),
                )
