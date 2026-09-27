"""Build and inspect disk-backed token datasets without corpus-sized RAM use."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from config.model_config import ModelConfig
from data.memmap_dataset import HEADER_SIZE, MAGIC, UINT32
from data.tokenizer import Tokenizer


SPLIT_FILES = {
    "base": {
        "train": Path("data/processed/train.txt"),
        "validation": Path("data/processed/validation.txt"),
        "test": Path("data/processed/test.txt"),
    },
    "healthcare": {
        "train": Path("data/processed/healthcare_train.txt"),
        "validation": Path("data/processed/healthcare_validation.txt"),
        "test": Path("data/processed/healthcare_test.txt"),
    },
    "continued_pretraining": {
        "train": Path("data/processed/continued_pretraining/train.txt"),
        "validation": Path("data/processed/continued_pretraining/validation.txt"),
        "test": Path("data/processed/continued_pretraining/test.txt"),
    },
}
TOKENIZER_FILE = Path("data/processed/tokenizer.json")
TOKEN_STORE_ROOT = Path("data/processed/token_store")
BASE_SPLIT_FILES = SPLIT_FILES["base"]
CORPUS_FILE = BASE_SPLIT_FILES["train"]

DEFAULT_CHUNK_CHARS = 1 << 20
DEFAULT_TEXT_BUFFER_CHARS = 1 << 20


def _atomic_replace(tmp: Path, target: Path) -> None:
    tmp.replace(target)


def _write_header(handle, length: int) -> None:
    import struct

    handle.write(struct.Struct("<8sQ").pack(MAGIC, int(length)))


def _iter_text_chunks(path: Path, chunk_chars: int = DEFAULT_CHUNK_CHARS):
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            chunk = handle.read(chunk_chars)
            if not chunk:
                break
            yield chunk


def _validate_paths(corpus_file: Path, tokenizer_file: Path) -> None:
    if not tokenizer_file.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_file}")
    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus split not found: {corpus_file}")


def build_memmap_token_store(
    corpus_file: Path,
    output_file: Path,
    tokenizer_file: Path | None = None,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
) -> dict:
    """Incrementally tokenize a text split into a compact uint32 disk store."""
    config = ModelConfig()
    tokenizer_path = Path(tokenizer_file) if tokenizer_file is not None else TOKENIZER_FILE
    corpus_file = Path(corpus_file)
    output_file = Path(output_file)
    _validate_paths(corpus_file, tokenizer_path)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    if len(tokenizer) != config.vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary ({len(tokenizer)}) does not match "
            f"model vocabulary ({config.vocab_size})"
        )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=output_file.name + ".",
        suffix=".tmp",
        dir=str(output_file.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)

    token_count = 0
    try:
        with tmp_path.open("wb") as out:
            out.seek(HEADER_SIZE)
            for chunk in _iter_text_chunks(corpus_file, chunk_chars):
                ids = tokenizer.encode(chunk)
                if ids:
                    import numpy as np

                    np.asarray(ids, dtype=UINT32).tofile(out)
                    token_count += len(ids)

            out.flush()
            os.fsync(out.fileno())

        with tmp_path.open("r+b") as out:
            out.seek(0)
            _write_header(out, token_count)
            out.flush()
            os.fsync(out.fileno())

        _atomic_replace(tmp_path, output_file)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return {
        "source": str(corpus_file),
        "output": str(output_file),
        "tokens": token_count,
        "bytes": output_file.stat().st_size,
    }


def token_store_path(dataset: str, split: str, output_root: Path | None = None) -> Path:
    if dataset not in SPLIT_FILES:
        raise ValueError(
            "Unknown dataset. Use base, healthcare, or continued_pretraining."
        )
    if split not in SPLIT_FILES[dataset]:
        raise ValueError(f"Unknown split: {split}. Use train, validation, or test.")
    root = Path(output_root) if output_root is not None else TOKEN_STORE_ROOT
    return root / dataset / f"{split}.tokens"


def prepare_dataset(
    dataset: str = "base",
    tokenizer_file: Path | None = None,
    output_root: Path | None = None,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
) -> list[dict]:
    """Prepare all available splits for one explicitly selected dataset."""
    results = []
    for split, corpus_file in SPLIT_FILES[dataset].items():
        output_file = token_store_path(dataset, split, output_root)
        try:
            result = build_memmap_token_store(
                corpus_file,
                output_file,
                tokenizer_file=tokenizer_file,
                chunk_chars=chunk_chars,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"{dataset}/{split}: unavailable ({exc})")
            continue
        results.append(result)
        print(f"{dataset}/{split}: {result['tokens']:,} tokens -> {result['output']}")
    return results


def load_token_ids_from_file(corpus_file: Path, tokenizer_file=None):
    """Backward-compatible eager loader for legacy callers."""
    config = ModelConfig()
    tokenizer_path = (
        Path(tokenizer_file) if tokenizer_file is not None else TOKENIZER_FILE
    )
    _validate_paths(Path(corpus_file), tokenizer_path)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    if len(tokenizer) != config.vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary ({len(tokenizer)}) does not match "
            f"model vocabulary ({config.vocab_size})"
        )
    text = Path(corpus_file).read_text(encoding="utf-8", errors="replace")
    token_ids = tokenizer.encode(text)
    if len(token_ids) <= config.context_length:
        raise ValueError(
            f"{corpus_file} does not contain enough tokens for the configured "
            f"context length ({config.context_length})."
        )
    return token_ids


def load_token_ids(split: str = "train", dataset: str = "base"):
    return load_token_ids_from_file(SPLIT_FILES[dataset][split])


def create_dataset(split: str = "train", dataset: str = "base"):
    from data.dataset import LanguageModelDataset

    config = ModelConfig()
    token_ids = load_token_ids(split, dataset=dataset)
    return LanguageModelDataset(
        token_ids=token_ids,
        context_length=config.context_length,
        stride=config.context_length,
    )


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Prepare one explicitly selected dataset as disk-backed token shards."
    )
    parser.add_argument("--dataset", choices=tuple(SPLIT_FILES), default="base")
    parser.add_argument(
        "--tokenizer-file", type=Path, default=TOKENIZER_FILE
    )
    parser.add_argument(
        "--output-root", type=Path, default=TOKEN_STORE_ROOT
    )
    parser.add_argument(
        "--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS
    )
    parsed = parser.parse_args(args)
    results = prepare_dataset(
        parsed.dataset,
        tokenizer_file=parsed.tokenizer_file,
        output_root=parsed.output_root,
        chunk_chars=parsed.chunk_chars,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
