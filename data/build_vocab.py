from collections import Counter
from pathlib import Path


CORPUS_FILE = Path("data/processed/corpus.txt")
VOCAB_FILE = Path("data/processed/vocab.txt")

VOCAB_SIZE = 10_000

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
]


def build_vocab() -> None:
    if not CORPUS_FILE.exists():
        raise FileNotFoundError(f"Corpus not found: {CORPUS_FILE}")

    text = CORPUS_FILE.read_text(
        encoding="utf-8",
        errors="replace",
    )

    words = text.split()
    counts = Counter(words)

    remaining_size = VOCAB_SIZE - len(SPECIAL_TOKENS)

    most_common = [
        token
        for token, _ in counts.most_common(remaining_size)
        if token not in SPECIAL_TOKENS
    ]

    vocabulary = SPECIAL_TOKENS + most_common

    VOCAB_FILE.write_text(
        "\n".join(vocabulary),
        encoding="utf-8",
    )

    print(f"Corpus words: {len(words):,}")
    print(f"Unique tokens: {len(counts):,}")
    print(f"Vocabulary size: {len(vocabulary):,}")
    print(f"Output: {VOCAB_FILE}")


if __name__ == "__main__":
    build_vocab()