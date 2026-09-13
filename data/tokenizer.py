from pathlib import Path


class Tokenizer:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

        self.token_to_id = {
            token: idx
            for idx, token in enumerate(vocabulary)
        }

        self.id_to_token = {
            idx: token
            for idx, token in enumerate(vocabulary)
        }

        required_tokens = {"<pad>", "<unk>", "<bos>", "<eos>"}

        missing = required_tokens - set(self.token_to_id)

        if missing:
            raise ValueError(
                f"Vocabulary missing special tokens: {missing}"
            )

    @classmethod
    def from_file(cls, vocab_path: str | Path):
        """Load a tokenizer vocabulary from a text file."""
        vocab_path = Path(vocab_path)

        if not vocab_path.exists():
            raise FileNotFoundError(
                f"Vocabulary file not found: {vocab_path}"
            )

        vocabulary = vocab_path.read_text(
            encoding="utf-8"
        ).splitlines()

        return cls(vocabulary)

    def encode(
        self,
        text,
        add_bos=False,
        add_eos=False,
    ):
        tokens = text.split()

        token_ids = [
            self.token_to_id.get(
                token,
                self.token_to_id["<unk>"],
            )
            for token in tokens
        ]

        if add_bos:
            token_ids.insert(
                0,
                self.token_to_id["<bos>"],
            )

        if add_eos:
            token_ids.append(
                self.token_to_id["<eos>"]
            )

        return token_ids

    def decode(self, token_ids):
        tokens = [
            self.id_to_token.get(
                token_id,
                "<unk>",
            )
            for token_id in token_ids
        ]

        return " ".join(tokens)

    def __len__(self):
        return len(self.vocabulary)