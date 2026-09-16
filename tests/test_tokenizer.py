from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from data.tokenizer import Tokenizer


def test_bpe_tokenizer_round_trip():
    tokenizer = HFTokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=64,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        min_frequency=1,
    )
    tokenizer.train_from_iterator(
        [
            "pneumonia is a respiratory infection",
            "respiratory infection may cause cough and fever",
            "medical terminology should be tokenized consistently",
        ],
        trainer,
    )

    wrapped = Tokenizer(tokenizer)
    original = "pneumonia infection"
    ids = wrapped.encode(original, add_bos=True, add_eos=True)

    assert ids[0] == wrapped.token_to_id["<bos>"]
    assert ids[-1] == wrapped.token_to_id["<eos>"]
    assert len(ids) > 2

    decoded = wrapped.decode(ids)
    body = decoded.replace("<bos>", "").replace("<eos>", "").strip()
    assert "infection" in body
    assert body.replace(" ", "") == original.replace(" ", "")
