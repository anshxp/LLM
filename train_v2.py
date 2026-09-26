"""Train the v2 encoder-decoder model from scratch, then SFT it."""

import argparse
import math
from itertools import islice
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config.seq2seq_model_config import Seq2SeqModelConfig
from data.seq2seq_dataset import BookSeq2SeqDataset, InstructionSeq2SeqDataset, collate_seq2seq
from data.tokenizer import Tokenizer
from model.encoder_decoder import EncoderDecoderLLM


def parse_args():
    parser = argparse.ArgumentParser(description="Train LLM v2 encoder-decoder Transformer.")
    parser.add_argument("--stage", choices=("pretrain", "sft"), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/v2"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/processed/tokenizer.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/v2"))
    parser.add_argument("--pretrained-checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=100)
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


def make_dataset(stage, split, args, config):
    root = args.data_root
    if stage == "pretrain":
        return BookSeq2SeqDataset(
            root / "books" / f"{split}.txt",
            tokenizer_path=args.tokenizer,
            context_length=config.context_length,
        )
    return InstructionSeq2SeqDataset(
        root / "sft" / f"{split}.jsonl",
        tokenizer_path=args.tokenizer,
        context_length=config.context_length,
    )


def loss_for_batch(model, batch, device):
    encoder_ids, decoder_ids, labels = batch
    logits = model(encoder_ids.to(device), decoder_ids.to(device))
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.to(device).reshape(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def evaluate(model, loader, device, max_batches):
    model.eval()
    losses = [loss_for_batch(model, batch, device).item() for batch in islice(loader, max_batches)]
    model.train()
    return sum(losses) / len(losses) if losses else math.inf


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
        raise ValueError("v2 currently uses batch_size=1 because attention padding masks are not enabled")
    if args.gradient_accumulation_steps < 1 or args.max_steps < 1:
        raise ValueError("gradient accumulation and max steps must be positive")

    torch.manual_seed(args.seed)
    device = device_from_arg(args.device)
    config = Seq2SeqModelConfig()
    tokenizer = Tokenizer.from_file(args.tokenizer)
    if len(tokenizer) != config.vocab_size:
        raise ValueError(f"Tokenizer vocabulary ({len(tokenizer)}) does not match model vocabulary ({config.vocab_size})")
    pad_id = tokenizer.token_to_id["<pad>"]

    model = EncoderDecoderLLM(config).to(device)
    if args.pretrained_checkpoint:
        if args.stage != "sft":
            raise ValueError("--pretrained-checkpoint is only used for the SFT stage")
        state = torch.load(args.pretrained_checkpoint, map_location=device, weights_only=False)
        state_dict = state.get("model_state_dict", state.get("model"))
        if state_dict is None:
            raise ValueError("Checkpoint does not contain model_state_dict or model")
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained v2 checkpoint: {args.pretrained_checkpoint}")
    elif args.stage == "sft":
        raise ValueError("SFT requires --pretrained-checkpoint from v2 pretraining")

    learning_rate = args.learning_rate or (2e-4 if args.stage == "pretrain" else 5e-5)
    train_dataset = make_dataset(args.stage, "train", args, config)
    validation_dataset = make_dataset(args.stage, "validation", args, config)
    loader_kwargs = {"batch_size": 1, "collate_fn": lambda batch: collate_seq2seq(batch, pad_id=pad_id)}
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, **loader_kwargs)

    print(f"Stage: {args.stage}")
    print(f"Device: {device}")
    print(f"Parameters: {model.num_parameters():,}")
    print(f"Effective batch size: {args.gradient_accumulation_steps}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_loss = math.inf
    train_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, args.max_steps + 1):
        accumulated = 0.0
        for _ in range(args.gradient_accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            loss = loss_for_batch(model, batch, device)
            (loss / args.gradient_accumulation_steps).backward()
            accumulated += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step == 1 or step % 25 == 0:
            print(f"step={step:,} train_loss={accumulated / args.gradient_accumulation_steps:.4f}")

        if step % args.eval_every == 0 or step == args.max_steps:
            validation_loss = evaluate(model, validation_loader, device, args.eval_batches)
            print(f"step={step:,} validation_loss={validation_loss:.4f}")
            save_checkpoint(args.checkpoint_dir / f"model_step_{step}.pt", model, optimizer, step, validation_loss)
            if validation_loss < best_loss:
                best_loss = validation_loss
                save_checkpoint(args.checkpoint_dir / "best_model.pt", model, optimizer, step, validation_loss)
                print("New best checkpoint saved.")

    print(f"Training complete. Best validation loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
