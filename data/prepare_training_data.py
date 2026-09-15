from pathlib import Path

from config.model_config import ModelConfig
from data.dataset import LanguageModelDataset
from data.tokenizer import Tokenizer


CORPUS_FILE = Path("data/processed/corpus.txt")
TOKENIZER_FILE = Path("data/processed/tokenizer.json")


def load_token_ids():
    config = ModelConfig()
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)

    if len(tokenizer) != config.vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary ({len(tokenizer)}) does not match "
            f"model vocabulary ({config.vocab_size})"
        )

    if not CORPUS_FILE.exists():
        raise FileNotFoundError(f"Corpus not found: {CORPUS_FILE}")

    text = CORPUS_FILE.read_text(
        encoding="utf-8",
        errors="replace",
    )
    token_ids = tokenizer.encode(text)

    if len(token_ids) <= config.context_length:
        raise ValueError(
            "Corpus does not contain enough tokens for the configured "
            f"context length ({config.context_length})."
        )

    return token_ids


def create_dataset():
    config = ModelConfig()
    token_ids = load_token_ids()

    return LanguageModelDataset(
        token_ids=token_ids,
        context_length=config.context_length,
    )


if __name__ == "__main__":
    dataset = create_dataset()
    print(f"Token IDs: {len(dataset.token_ids):,}")
    print(f"Context length: {dataset.context_length}")
    print(f"Training examples: {len(dataset):,}")

    input_ids, target_ids = dataset[0]
    print(f"Input shape: {tuple(input_ids.shape)}")
    print(f"Target shape: {tuple(target_ids.shape)}")
