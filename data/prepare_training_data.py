from pathlib import Path

from config.model_config import ModelConfig
from data.dataset import LanguageModelDataset
from data.tokenizer import Tokenizer


SPLIT_FILES = {
    "train": Path("data/processed/train.txt"),
    "validation": Path("data/processed/validation.txt"),
    "test": Path("data/processed/test.txt"),
}
# Backward-compatible default corpus path used by tests and callers that need
# to inspect the primary training corpus before split files are generated.
CORPUS_FILE = SPLIT_FILES["train"]
TOKENIZER_FILE = Path("data/processed/tokenizer.json")


def load_token_ids(split: str = "train"):
    """Load token IDs for one corpus split using the trained tokenizer."""
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown split: {split}. Use train, validation, or test.")

    config = ModelConfig()
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)

    if len(tokenizer) != config.vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary ({len(tokenizer)}) does not match "
            f"model vocabulary ({config.vocab_size})"
        )

    corpus_file = SPLIT_FILES[split]
    if split == "train" and CORPUS_FILE != SPLIT_FILES["train"]:
        corpus_file = CORPUS_FILE

    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus split not found: {corpus_file}")

    text = corpus_file.read_text(encoding="utf-8", errors="replace")
    token_ids = tokenizer.encode(text)

    if len(token_ids) <= config.context_length:
        raise ValueError(
            f"{split} split does not contain enough tokens for the configured "
            f"context length ({config.context_length})."
        )

    return token_ids


def create_dataset(split: str = "train"):
    config = ModelConfig()
    token_ids = load_token_ids(split)

    return LanguageModelDataset(
        token_ids=token_ids,
        context_length=config.context_length,
        stride=config.context_length,
    )


if __name__ == "__main__":
    for split in SPLIT_FILES:
        try:
            dataset = create_dataset(split)
        except ValueError as exc:
            print(f"{split}: unavailable ({exc})")
            continue
        print(f"{split}: {len(dataset.token_ids):,} tokens, {len(dataset):,} sequences")
