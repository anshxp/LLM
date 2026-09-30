from pathlib import Path

from config.model_config import ModelConfig
from data.dataset import StreamingLanguageModelDataset
from data.parquet_dataset import ParquetLanguageModelDataset
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
STAGE1_PARQUET = Path("data/raw/stage 1/train.parquet")
TOKENIZER_FILE = Path("data/processed/tokenizer.json")
BASE_SPLIT_FILES = SPLIT_FILES["base"]
CORPUS_FILE = BASE_SPLIT_FILES["train"]

# Keep the legacy in-memory API useful for small evaluation/debugging corpora,
# while preventing it from accidentally loading large training corpora into RAM.
LEGACY_IN_MEMORY_LIMIT_BYTES = 64 * 1024 * 1024


def resolve_corpus_file(split: str = "train", dataset: str = "base") -> Path:
    """Resolve a text corpus path for legacy/base and healthcare datasets."""
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
    """Load token IDs for small legacy/evaluation callers.

    Training must use ``create_dataset`` so multi-GB corpora remain disk-backed.
    """
    config = ModelConfig()
    corpus_file = resolve_corpus_file(split, dataset=dataset)

    if not TOKENIZER_FILE.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {TOKENIZER_FILE}")

    if corpus_file.stat().st_size > LEGACY_IN_MEMORY_LIMIT_BYTES:
        raise RuntimeError(
            f"{corpus_file} is larger than the {LEGACY_IN_MEMORY_LIMIT_BYTES // (1024 * 1024)} MB "
            "legacy in-memory limit. Use create_dataset(), which streams directly from disk."
        )

    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
    if len(tokenizer) != config.vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary ({len(tokenizer)}) does not match "
            f"model vocabulary ({config.vocab_size})"
        )

    text = corpus_file.read_text(encoding="utf-8", errors="replace")
    token_ids = tokenizer.encode(text)
    if len(token_ids) <= config.context_length:
        raise ValueError(
            f"{corpus_file} does not contain enough tokens for the configured "
            f"context length ({config.context_length})."
        )
    return token_ids


def _load_tokenizer(config: ModelConfig) -> Tokenizer:
    if not TOKENIZER_FILE.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {TOKENIZER_FILE}")
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
    if len(tokenizer) != config.vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary ({len(tokenizer)}) does not match "
            f"model vocabulary ({config.vocab_size})"
        )
    return tokenizer


def create_dataset(
    split: str = "train",
    dataset: str = "base",
    stride: int | None = None,
    stage1_parquet: Path = STAGE1_PARQUET,
):
    """Create a disk-backed dataset that tokenizes records incrementally.

    ``dataset=stage1`` streams the Parquet corpus directly and uses the existing
    Stage 0 tokenizer. It does not retrain or replace the tokenizer.
    """
    config = ModelConfig()
    tokenizer = _load_tokenizer(config)

    if stride is None:
        stride = 128 if split == "train" else 256

    if dataset == "stage1":
        if not stage1_parquet.exists():
            raise FileNotFoundError(f"Stage 1 Parquet corpus not found: {stage1_parquet}")
        return ParquetLanguageModelDataset(
            parquet_file=stage1_parquet,
            tokenizer=tokenizer,
            context_length=config.context_length,
            stride=stride,
            split=split,
        )

    corpus_file = resolve_corpus_file(split, dataset=dataset)
    return StreamingLanguageModelDataset(
        corpus_file=corpus_file,
        tokenizer=tokenizer,
        context_length=config.context_length,
        stride=stride,
    )


if __name__ == "__main__":
    for dataset in (*SPLIT_FILES, "stage1"):
        for split in ("train", "validation", "test"):
            try:
                dataset_obj = create_dataset(split, dataset=dataset)
            except (FileNotFoundError, ValueError) as exc:
                print(f"{dataset}/{split}: unavailable ({exc})")
                continue
            corpus_name = getattr(dataset_obj, "corpus_file", getattr(dataset_obj, "parquet_file", "unknown"))
            print(
                f"{dataset}/{split}: streaming from {corpus_name} "
                f"(context={dataset_obj.context_length}, stride={dataset_obj.stride})"
            )
