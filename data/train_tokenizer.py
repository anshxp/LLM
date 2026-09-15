from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


CORPUS_FILE = Path("data/processed/corpus.txt")
TOKENIZER_FILE = Path("data/processed/tokenizer.json")
VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def train_tokenizer() -> None:
    if not CORPUS_FILE.exists():
        raise FileNotFoundError(f"Corpus not found: {CORPUS_FILE}")

    TOKENIZER_FILE.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
    )

    tokenizer.train([str(CORPUS_FILE)], trainer)
    tokenizer.save(str(TOKENIZER_FILE))

    print(f"Vocabulary size: {tokenizer.get_vocab_size():,}")
    print(f"Output: {TOKENIZER_FILE}")


if __name__ == "__main__":
    train_tokenizer()
