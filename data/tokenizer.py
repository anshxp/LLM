from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


class Tokenizer:
    """Thin wrapper around a trained BPE tokenizer."""

    def __init__(self, tokenizer: HFTokenizer):
        self._tokenizer = tokenizer
        self.token_to_id = {
            token: tokenizer.token_to_id(token)
            for token in SPECIAL_TOKENS
        }

        missing = [
            token for token, token_id in self.token_to_id.items()
            if token_id is None
        ]
        if missing:
            raise ValueError(
                f"Tokenizer missing special tokens: {missing}"
            )

    @classmethod
    def from_file(cls, tokenizer_path: str | Path):
        tokenizer_path = Path(tokenizer_path)
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer file not found: {tokenizer_path}"
            )
        return cls(HFTokenizer.from_file(str(tokenizer_path)))

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False):
        encoding = self._tokenizer.encode(text)
        token_ids = list(encoding.ids)

        if add_bos:
            token_ids.insert(0, self.token_to_id["<bos>"])
        if add_eos:
            token_ids.append(self.token_to_id["<eos>"])

        return token_ids

    def decode(self, token_ids):
        return self._tokenizer.decode(token_ids, skip_special_tokens=False)

    def __len__(self):
        return self._tokenizer.get_vocab_size()
