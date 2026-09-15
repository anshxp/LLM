from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


TRAIN_FILE = Path("data/processed/train.txt")
TOKENIZER_FILE = Path("data/processed/tokenizer.json")
VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def train_tokenizer() -> None:
    """Train the model tokenizer using training data only."""
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training corpus not found: {TRAIN_FILE}")

    TOKENIZER_FILE.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
    )

    tokenizer.train([str(TRAIN_FILE)], trainer)
    actual_vocab_size = tokenizer.get_vocab_size()
    if actual_vocab_size != VOCAB_SIZE:
        raise ValueError(
            f"Corpus produced vocabulary size {actual_vocab_size}; "
            f"expected {VOCAB_SIZE}. Add more training data or lower "
            "VOCAB_SIZE."
        )

    tokenizer.save(str(TOKENIZER_FILE))
    print(f"Vocabulary size: {actual_vocab_size:,}")
    print(f"Output: {TOKENIZER_FILE}")


if __name__ == "__main__":
    train_tokenizer()
