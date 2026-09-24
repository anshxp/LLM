import json

import torch
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from data.finetuning2_pipeline import records_from_object, split_name, token_stats
from data.tokenizer import Tokenizer
from training.loss import language_model_loss


def test_question_answer_normalizes_to_sft_record():
    records = list(records_from_object({"question": "What is anemia?", "answer": "A condition involving reduced oxygen-carrying capacity."}, "qa.json", "1"))
    assert len(records) == 1
    record = records[0]
    assert record["instruction"] == "Answer the question accurately."
    assert record["input"] == "What is anemia?"
    assert record["response"].startswith("A condition")
    assert record["source"] == "qa.json"
    assert record["source_id"] == "1"


def test_conversation_normalizes_to_sft_record():
    obj = {"messages": [
        {"role": "user", "content": "Explain anemia."},
        {"role": "assistant", "content": "Anemia is a condition involving too few healthy red blood cells."},
    ]}
    records = list(records_from_object(obj, "conversation.jsonl", "3"))
    assert len(records) == 1
    assert records[0]["input"] == "Explain anemia."
    assert records[0]["response"].startswith("Anemia is")


def test_split_is_deterministic():
    assert split_name("00000000000000000000000000000000") == "train"
    assert split_name("ffffffffffffffffffffffffffffffff") == "test"
    assert split_name("0000000000000000cccccccccccccccc") == split_name("0000000000000000cccccccccccccccc")


def test_token_stats_uses_existing_tokenizer(tmp_path):
    path = tmp_path / "tokenizer.json"
    tokenizer = HFTokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=64, special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"], min_frequency=1)
    tokenizer.train_from_iterator(["### Instruction: Answer the question accurately.", "### Input: anemia", "### Response: A condition."], trainer=trainer)
    tokenizer.save(str(path))
    wrapper = Tokenizer.from_file(path)
    stats = token_stats({"instruction": "Answer the question accurately.", "input": "anemia", "response": "A condition."}, wrapper, 256)
    assert stats["total_tokens"] > 0
    assert stats["fits_context"] is True
    assert stats["unk_tokens"] >= 0


def test_masked_targets_are_ignored_by_sft_loss():
    logits = torch.tensor([[[4.0, 0.0], [0.0, 4.0]]])
    targets = torch.tensor([[-100, 1]])
    loss = language_model_loss(logits, targets)
    expected = torch.nn.functional.cross_entropy(logits[:, 1:, :].reshape(-1, 2), torch.tensor([1]))
    assert torch.allclose(loss, expected)
