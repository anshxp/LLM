from pathlib import Path

from config.model_config import ModelConfig
from data.memmap_dataset import HEADER_SIZE, MAGIC, TOKEN_DTYPE
from data.tokenizer import Tokenizer
import argparse
import json
import os
import struct
import tempfile

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
HEADER_STRUCT = struct.Struct("<8sQ")


def _iter_text_chunks(path: Path, chunk_chars: int):
    """Yield exact-text chunks whose non-final boundaries are whitespace.

    A boundary is selected at the last whitespace at or before the requested
    chunk size. The boundary whitespace stays in the emitted chunk. If no
    whitespace is available because a single token crosses the limit, the
    token is held until whitespace is encountered; that whitespace is retained
    as the first character of the following chunk. This keeps ordinary words
    intact while also satisfying the long-token boundary contract.
    """
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        pending = ""
        oversize_token = False

        while True:
            piece = handle.read(chunk_chars)
            if not piece:
                if pending:
                    yield pending
                return

            pending += piece

            # If the buffer begins with an oversized non-whitespace token,
            # locate its first whitespace. Emit the token separately and keep
            # the delimiter for the next chunk.
            if oversize_token:
                boundary = next(
                    (i for i, char in enumerate(pending) if char.isspace()),
                    None,
                )
                if boundary is None:
                    continue
                yield pending[:boundary]
                pending = pending[boundary:]
                oversize_token = False
                if not pending:
                    continue

            # Keep a normal chunk within the requested size where possible.
            if len(pending) < chunk_chars:
                continue

            boundary = None
            for index in range(min(chunk_chars, len(pending)) - 1, -1, -1):
                if pending[index].isspace():
                    boundary = index + 1
                    break

            if boundary is None:
                # No whitespace before the limit: this is an oversized token.
                # Hold it intact and continue until its terminating whitespace.
                oversize_token = True
                continue

            yield pending[:boundary]
            pending = pending[boundary:]


def _atomic_replace(tmp_path: Path, target: Path) -> None:
    tmp_path.replace(target)


def _write_header(path: Path, token_count: int) -> None:
    with path.open("r+b") as handle:
        handle.seek(0)
        handle.write(HEADER_STRUCT.pack(MAGIC, int(token_count)))
        handle.flush()
        os.fsync(handle.fileno())


def token_store_path(dataset: str, split: str, output_root: Path | None = None) -> Path:
    if dataset not in SPLIT_FILES:
        raise ValueError(
            "Unknown dataset. Use base, healthcare, or continued_pretraining."
        )
    if split not in SPLIT_FILES[dataset]:
        raise ValueError(f"Unknown split: {split}. Use train, validation, or test.")
    root = Path(output_root) if output_root is not None else TOKEN_STORE_ROOT
    return root / dataset / f"{split}.tokens"


def build_memmap_token_store(
    corpus_file: Path,
    output_file: Path,
    tokenizer_file: Path | None = None,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
) -> dict:
    config = ModelConfig()
    corpus_file = Path(corpus_file)
    tokenizer_path = Path(tokenizer_file) if tokenizer_file is not None else TOKENIZER_FILE

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus split not found: {corpus_file}")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    if len(tokenizer) != config.vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary ({len(tokenizer)}) does not match "
            f"model vocabulary ({config.vocab_size})"
        )

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=output_file.name + ".",
        suffix=".tmp",
        dir=str(output_file.parent),
    )
    os.close(fd)
    tmp_path = Path(name)
    token_count = 0

    try:
        with tmp_path.open("wb") as handle:
            handle.seek(HEADER_SIZE)
            for chunk in _iter_text_chunks(corpus_file, chunk_chars):
                token_ids = tokenizer.encode(chunk)
                if token_ids:
                    import numpy as np

                    np.asarray(token_ids, dtype=TOKEN_DTYPE).tofile(handle)
                    token_count += len(token_ids)
            handle.flush()
            os.fsync(handle.fileno())

        _write_header(tmp_path, token_count)
        _atomic_replace(tmp_path, output_file)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return {
        "source": str(corpus_file),
        "output": str(output_file),
        "tokens": token_count,
        "bytes": output_file.stat().st_size,
        "chunk_chars": chunk_chars,
    }


def prepare_dataset(
    dataset: str = "base",
    tokenizer_file: Path | None = None,
    output_root: Path | None = None,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
) -> list[dict]:
    if dataset not in SPLIT_FILES:
        raise ValueError(
            "Unknown dataset. Use base, healthcare, or continued_pretraining."
        )
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
    """Legacy eager API retained for callers/tests; large training must use memmap stores."""
    config = ModelConfig()
    tokenizer_path = Path(tokenizer_file) if tokenizer_file is not None else TOKENIZER_FILE
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    if len(tokenizer) != config.vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary ({len(tokenizer)}) does not match "
            f"model vocabulary ({config.vocab_size})"
        )
    corpus_file = Path(corpus_file)
    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus split not found: {corpus_file}")
    text = corpus_file.read_text(encoding="utf-8", errors="replace")
    token_ids = tokenizer.encode(text)
    if len(token_ids) <= config.context_length:
        raise ValueError(
            f"{corpus_file} does not contain enough tokens for the configured "
            f"context length ({config.context_length})."
        )
    return token_ids


def load_token_ids(split: str = "train", dataset: str = "base"):
    if dataset not in SPLIT_FILES or split not in SPLIT_FILES[dataset]:
        raise ValueError(f"Unknown dataset/split: {dataset}/{split}")
    return load_token_ids_from_file(SPLIT_FILES[dataset][split])


def create_dataset(split: str = "train", dataset: str = "base"):
    from data.dataset import LanguageModelDataset

    config = ModelConfig()
    token_ids = load_token_ids(split, dataset=dataset)
    return LanguageModelDataset(token_ids, config.context_length, config.context_length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(SPLIT_FILES), default="base")
    parser.add_argument("--tokenizer-file", type=Path, default=TOKENIZER_FILE)
    parser.add_argument("--output-root", type=Path, default=TOKEN_STORE_ROOT)
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    args = parser.parse_args()
    print(json.dumps(
        prepare_dataset(
            args.dataset,
            tokenizer_file=args.tokenizer_file,
            output_root=args.output_root,
            chunk_chars=args.chunk_chars,
        ),
        indent=2,
    ))
