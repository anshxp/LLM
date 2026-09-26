"""Train v2 from scratch on the streaming raw pretraining corpus.

SFT is intentionally unavailable in this training entry point until the
pretraining phase has produced a checkpoint and the user explicitly starts
that phase later.
"""

import argparse
import math
from itertools import islice
from pathlib import Path

import torch
import torch.nn.functional as F

from config.seq2seq_model_config import Seq2SeqModelConfig
from data.corpus_pipeline import iter_pretraining_text
from data.tokenizer import Tokenizer
from model.encoder_decoder import EncoderDecoderLLM


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain LLM v2 encoder-decoder Transformer.")
    parser.add_argument("--stage", choices=("pretrain",), default="pretrain")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/processed/tokenizer.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/v2"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def device_from_arg(value):
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _encode_document(text, tokenizer, context_length):
    ids = tokenizer.encode(text).ids
    if len(ids) < 2:
        return []
    # Encoder receives a prefix; decoder predicts the continuation.
    examples = []
    stride = context_length
    for start in range(0, len(ids) - 1, stride):
        chunk = ids[start:start + context_length + 1]
        if len(chunk) < 2:
            break
        split = max(1, min(context_length // 2, len(chunk) - 1))
        encoder_ids = chunk[:split]
        target = chunk[split:]
        if not target:
            continue
        decoder_input = [tokenizer.token_to_id["<bos>"]] + target[:-1]
        labels = target
        examples.append((encoder_ids, decoder_input, labels))
    return examples


def _pad_batch(examples, pad_id):
    max_enc = max(len(x[0]) for x in examples)
    max_dec = max(len(x[1]) for x in examples)
    enc, dec, labels = [], [], []
    for e, d, y in examples:
        enc.append(e + [pad_id] * (max_enc - len(e)))
        dec.append(d + [pad_id] * (max_dec - len(d)))
        labels.append(y + [-100] * (max_dec - len(y)))
    return torch.tensor(enc, dtype=torch.long), torch.tensor(dec, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def stream_examples(raw_root, tokenizer, context_length):
    for text, source in iter_pretraining_text(raw_root):
        for example in _encode_document(text, tokenizer, context_length):
            yield example


def next_batch(stream, tokenizer, context_length, pad_id, batch_size=1):
    examples = []
    while len(examples) < batch_size:
        try:
            examples.append(next(stream))
        except StopIteration:
            return None
    return _pad_batch(examples, pad_id)


def loss_for_batch(model, batch, device):
    encoder_ids, decoder_ids, labels = batch
    logits = model(encoder_ids.to(device), decoder_ids.to(device))
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.to(device).reshape(-1), ignore_index=-100
    )


def save_checkpoint(path, model, optimizer, step, validation_loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "validation_loss": validation_loss,
    }, path)


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("The current streaming pretraining path uses batch_size=1")
    if args.gradient_accumulation_steps < 1 or args.max_steps < 1:
        raise ValueError("gradient accumulation and max steps must be positive")

    torch.manual_seed(args.seed)
    device = device_from_arg(args.device)
    config = Seq2SeqModelConfig()
    tokenizer = Tokenizer.from_file(args.tokenizer)
    if len(tokenizer) != config.vocab_size:
        raise ValueError(f"Tokenizer vocabulary ({len(tokenizer)}) does not match model vocabulary ({config.vocab_size})")
    pad_id = tokenizer.token_to_id["<pad>"]

    # Fresh initialization is intentional for v2; no v1 checkpoint is loaded.
    model = EncoderDecoderLLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("Stage: pretrain")
    print(f"Raw corpus: {args.raw_root}")
    print(f"Device: {device}")
    print(f"Parameters: {model.num_parameters():,}")
    print("Corpus mode: streaming; no processed corpus copy is created")
    print(f"Effective batch size: {args.gradient_accumulation_steps}")

    best_loss = math.inf
    stream = iter(stream_examples(args.raw_root, tokenizer, config.context_length))
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, args.max_steps + 1):
        accumulated = 0.0
        for _ in range(args.gradient_accumulation_steps):
            batch = next_batch(stream, tokenizer, config.context_length, pad_id, args.batch_size)
            if batch is None:
                stream = iter(stream_examples(args.raw_root, tokenizer, config.context_length))
                batch = next_batch(stream, tokenizer, config.context_length, pad_id, args.batch_size)
            loss = loss_for_batch(model, batch, device)
            (loss / args.gradient_accumulation_steps).backward()
            accumulated += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step == 1 or step % 25 == 0:
            print(f"step={step:,} train_loss={accumulated / args.gradient_accumulation_steps:.4f}")

        # Save periodic checkpoints. A separate validation corpus is intentionally
        # not materialized yet; validation will be added without duplicating the
        # raw corpus when the pretraining run is stabilized.
        if step % args.eval_every == 0 or step == args.max_steps:
            save_checkpoint(args.checkpoint_dir / f"model_step_{step}.pt", model, optimizer, step, math.inf)
            save_checkpoint(args.checkpoint_dir / "latest_model.pt", model, optimizer, step, math.inf)
            print(f"checkpoint saved at step={step:,}")

    print("Pretraining complete. SFT is intentionally not run by this command.")


if __name__ == "__main__":
    main()
