from pathlib import Path

from config.model_config import ModelConfig
from data.dataset import StreamingLanguageModelDataset
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
}
TOKENIZER_FILE = Path("data/processed/tokenizer.json")
BASE_SPLIT_FILES = SPLIT_FILES["base"]
CORPUS_FILE = BASE_SPLIT_FILES["train"]


def resolve_corpus_file(split: str = "train", dataset: str = "base") -> Path:
    """Resolve the corpus path without loading its contents into RAM."""
    if dataset not in SPLIT_FILES:
        raise ValueError(f"Unknown dataset: {dataset}. Use base or healthcare.")
    if split not in SPLIT_FILES[dataset]:
        raise ValueError(f"Unknown split: {split}. Use train, validation, or test.")

    corpus_file = SPLIT_FILES[dataset][split]
    if dataset == "base" and split == "train" and CORPUS_FILE != BASE_SPLIT_FILES["train"]:
        corpus_file = CORPUS_FILE
    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus split not found: {corpus_file}")
    return corpus_file


def load_token_ids(split: str = "train", dataset: str = "base"):
    """Deprecated in-memory loader kept for compatibility.

    Large corpora should use ``create_dataset`` below, which streams directly
    from disk. This function intentionally raises instead of silently loading
    a multi-GB corpus into RAM.
    """
    raise RuntimeError(
        "load_token_ids() is disabled for large-corpus training. "
        "Use create_dataset(), which returns a disk-backed streaming dataset."
    )


def create_dataset(split: str = "train", dataset: str = "base", stride: int | None = None):
    """Create a disk-backed dataset that tokenizes records incrementally."""
    config = ModelConfig()
    corpus_file = resolve_corpus_file(split, dataset=dataset)
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)

    if len(tokenizer) != config.vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary ({len(tokenizer)}) does not match "
            f"model vocabulary ({config.vocab_size})"
        )

    if stride is None:
        stride = config.context_length

    return StreamingLanguageModelDataset(
        corpus_file=corpus_file,
        tokenizer=tokenizer,
        context_length=config.context_length,
        stride=stride,
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
                f"{dataset}/{split}: streaming from {dataset_obj.corpus_file} "
                f"(context={dataset_obj.context_length}, stride={dataset_obj.stride})"
            )
