from pathlib import Path

from config.model_config import ModelConfig
from data.dataset import LanguageModelDataset
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
BASE_SPLIT_FILES = SPLIT_FILES["base"]
CORPUS_FILE = BASE_SPLIT_FILES["train"]


def load_token_ids_from_file(corpus_file: Path, tokenizer_file: Path = TOKENIZER_FILE):
    """Tokenize one already-built corpus file with the project tokenizer."""
    config = ModelConfig()
    tokenizer = Tokenizer.from_file(tokenizer_file)

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
    """Load token IDs for one corpus split using the trained tokenizer."""
    if dataset not in SPLIT_FILES:
        raise ValueError(
            "Unknown dataset. Use base, healthcare, or continued_pretraining."
        )
    if split not in SPLIT_FILES[dataset]:
        raise ValueError(f"Unknown split: {split}. Use train, validation, or test.")

    corpus_file = SPLIT_FILES[dataset][split]
    if dataset == "base" and split == "train" and CORPUS_FILE != BASE_SPLIT_FILES["train"]:
        corpus_file = CORPUS_FILE
    return load_token_ids_from_file(corpus_file)


def create_dataset(split: str = "train", dataset: str = "base"):
    config = ModelConfig()
    token_ids = load_token_ids(split, dataset=dataset)

    return LanguageModelDataset(
        token_ids=token_ids,
        context_length=config.context_length,
        stride=config.context_length,
    )


if __name__ == "__main__":
    for dataset in SPLIT_FILES:
        for split in SPLIT_FILES[dataset]:
            try:
                dataset_obj = create_dataset(split, dataset=dataset)
            except (FileNotFoundError, ValueError) as exc:
                print(f"{dataset}/{split}: unavailable ({exc})")
                continue
            print(
                f"{dataset}/{split}: {len(dataset_obj.token_ids):,} tokens, "
                f"{len(dataset_obj):,} sequences"
            )
