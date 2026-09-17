import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


TRAIN_FILES = {
    "base": Path("data/processed/train.txt"),
    "healthcare": Path("data/processed/healthcare_train.txt"),
}
TOKENIZER_FILE = Path("data/processed/tokenizer.json")
VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def train_tokenizer(dataset: str = "base", vocab_size: int = VOCAB_SIZE) -> None:
    """Train the project BPE tokenizer using one training corpus only."""
    if dataset not in TRAIN_FILES:
        raise ValueError("Unknown dataset. Use base or healthcare.")
    if vocab_size <= len(SPECIAL_TOKENS):
        raise ValueError("vocab_size must be larger than the number of special tokens")

    train_file = TRAIN_FILES[dataset]
    if not train_file.exists():
        raise FileNotFoundError(f"Training corpus not found: {train_file}")

    TOKENIZER_FILE.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
    )

    tokenizer.train([str(train_file)], trainer)
    actual_vocab_size = tokenizer.get_vocab_size()
    if actual_vocab_size != vocab_size:
        raise ValueError(
            f"Corpus produced vocabulary size {actual_vocab_size}; "
            f"expected {vocab_size}. Add more training data or request a lower "
            "vocab size."
        )

    tokenizer.save(str(TOKENIZER_FILE))
    print(f"Dataset: {dataset}")
    print(f"Vocabulary size: {actual_vocab_size:,}")
    print(f"Output: {TOKENIZER_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the project BPE tokenizer.")
    parser.add_argument("--dataset", choices=tuple(TRAIN_FILES), default="base")
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    args = parser.parse_args()
    train_tokenizer(args.dataset, vocab_size=args.vocab_size)
